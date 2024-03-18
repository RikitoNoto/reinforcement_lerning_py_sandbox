import gymnasium as gym
import time
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# from stable_baselines3.common.utils import set_global_seeds
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    ClipRewardEnv,
    EpisodicLifeEnv,
)
from callbacks import ModelSaveCallback

# 定数
ENV_ID = "BreakoutNoFrameskip-v4"  # 環境ID
NUM_ENV = 64  # 環境数


# 環境を生成する関数
def make_env(env_id, rank, render_mode="human", seed=0, episodic_life=True):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = NoopResetEnv(
            env, noop_max=30
        )  # 環境リセット後の数ステップ間の行動「Noop」
        env = MaxAndSkipEnv(env, skip=4)  # 4フレームごとに行動を選択
        env = FireResetEnv(env)  # 環境リセット後の行動「Fire」
        env = WarpFrame(env)  # 画面イメージを84x84のグレースケールに変換
        # env = ScaledFloatFrame(env)  # 状態の正規化
        env = ClipRewardEnv(env)  # 報酬の「-1」「0」「1」クリッピング
        if episodic_life:
            env = EpisodicLifeEnv(env)  # ライフ1減でエピソード完了

        if rank == 0:
            os.makedirs("logs", exist_ok=True)
            env = Monitor(env, "logs/", allow_early_resets=True)
        # env.seed(seed + rank)
        return env

    # set_global_seeds(seed)
    return _init


# メイン関数の定義
def learn():
    # 学習環境の生成
    train_env = DummyVecEnv(
        [make_env(ENV_ID, i, render_mode=None) for i in range(NUM_ENV - 1)]
    )

    # モデルの生成
    model = PPO("CnnPolicy", train_env, verbose=0)

    # モデルの読み込み
    if os.path.exists("saves/breakout_model.zip"):
        model = PPO.load("saves/breakout_model", env=train_env, verbose=0)
    model_save_callback = ModelSaveCallback(
        "logs", "saves", save_epoch=1000, overwrite=False
    )
    # モデルの学習
    # model.learn(
    #     total_timesteps=1280000, callback=model_save_callback, progress_bar=True
    # )
    model.learn(
        total_timesteps=60000000, callback=model_save_callback, progress_bar=True
    )


def play():
    # テスト環境の生成
    test_env = DummyVecEnv([make_env(ENV_ID, 9, episodic_life=False)])

    # モデルの生成
    model = PPO("CnnPolicy", test_env, verbose=0, clip_range=0.1)

    # モデルの読み込み
    model = PPO.load("saves/breakout_model", env=test_env, verbose=0)
    # モデルのテスト
    state = test_env.reset()
    total_reward = 0
    while True:
        # 環境の描画
        test_env.render()
        time.sleep(1 / 60)

        # モデルの推論
        action, _ = model.predict(state)

        # 1ステップ実行
        state, reward, done, info = test_env.step(action)

        # エピソードの完了
        total_reward += reward[0]
        if done:
            print("reward:", total_reward)
            state = test_env.reset()
            total_reward = 0


# メインの実行
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "play":

        play()
    else:
        learn()
