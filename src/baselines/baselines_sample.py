import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# 環境の生成
env = gym.make("CartPole-v1", render_mode="human")
# マルチプロセスで動作させるかどうか。マルチプロセスはSubprocVecEnv
env = DummyVecEnv([lambda: env])

# 方策をMlpPolicyでモデル生成
model = PPO("MlpPolicy", env, verbose=1)

# モデルの学習
model.learn(total_timesteps=128000)

# モデルのテスト
state = env.reset()
while True:
    # 環境の描画
    env.render()

    # モデルの推論 (4)
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, done, info = env.step(action)

    # エピソード完了
    if done:
        break

# 環境のクローズ
env.close()
