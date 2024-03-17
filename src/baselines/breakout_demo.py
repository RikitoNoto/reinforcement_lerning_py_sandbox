import random
import pyglet
import gymnasium as gym
import time
from pyglet.window import key
from imitation.data.types import Trajectory
import pickle
from stable_baselines3.common.atari_wrappers import *

# 環境の生成
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = MaxAndSkipEnv(env, skip=4)  # 4フレームごとに行動を選択
env = WarpFrame(env)  # 画面イメージを84x84のグレースケールに変換
state = env.reset()
env.render()

# キーイベント用のウィンドウの生成
win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()
win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()


# キー状態の取得
def get_key_state():
    key_state = set()
    win.dispatch_events()

    for key_code, pressed in key_handler.data.items():
        if pressed:
            key_state.add(key_code)
    return key_state


# キー入力待ち
while len(get_key_state()) == 0:
    time.sleep(1.0 / 30.0)


# 人間のデモを収集するコールバック
def human_expert(_state):
    # キー状態の取得
    key_state = get_key_state()

    # 行動の選択
    action = 0
    if key.SPACE in key_state:
        action = 1
    elif key.RIGHT in key_state:
        action = 2
    elif key.LEFT in key_state:
        action = 3

    # スリープ
    time.sleep(1.0 / 60.0)

    # 環境の描画
    env.render()

    # 行動の選択
    return action


actions = []
infos = []
observations = [state[0]]
while True:
    env.render()
    action = human_expert(state)
    state, reward, done, truncated, info = env.step(action)
    actions.append(action)
    infos.append(info)
    observations.append(state)

    if done:
        ts = Trajectory(
            obs=np.array(observations),
            acts=np.array(actions),
            infos=np.array(infos),
            terminal=True,
        )
        with open("breakout.pickle", mode="wb") as f:
            pickle.dump([ts], f)
        break
