import asyncio
import websockets
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time

ACTION_KEYS = [
    [],
    ['left'],
    ['right'],
    ['jump'],
    ['down'],
    ['att'],
    ['def'],
    ['left', 'att'],
    ['right', 'att'],
    ['jump', 'att'],
    ['down', 'att'],
]

OBS_SCALE = np.array([
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0,
], dtype=np.float32)

OBS_OFFSET = np.array([
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
], dtype=np.float32)


class LF2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)
        self.ws = None
        self.latest_state = None
        self.prev_state = None
        self._lock = threading.Lock()
        self._episode = 0
        self._start_server()

    def _start_server(self):
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, "localhost", 8765):
            await asyncio.Future()

    async def _handler(self, websocket):
        self.ws = websocket
        print("游戏已连接！")
        async for message in websocket:
            data = json.loads(message)
            with self._lock:
                self.latest_state = data

    def _send_action(self, action):
        if self.ws is None:
            return
        keys = ACTION_KEYS[action]
        msg = json.dumps({"action": int(action), "keys": keys})
        asyncio.run_coroutine_threadsafe(
            self.ws.send(msg), self.loop
        )

    def _state_to_obs(self, state):
        d = state['davis']
        e = state['dennis']
        dir_map = {'right': 1.0, 'left': -1.0}
        dx = d['x'] - e['x']
        raw = np.array([
            d['x'], d['y'], d['hp'], d['mp'],
            float(d['frame']), dir_map.get(d['dir'], 0.0),
            e['x'], e['y'], e['hp'], e['mp'],
            float(e['frame']), dir_map.get(e['dir'], 0.0),
            dx,
        ], dtype=np.float32)
        obs = raw / OBS_SCALE - OBS_OFFSET
        return np.clip(obs, -2.0, 2.0)

    def _compute_reward(self, prev, curr):
        if prev is None:
            return 0.0

        d_hp_delta = curr['davis']['hp'] - prev['davis']['hp']
        e_hp_delta = curr['dennis']['hp'] - prev['dennis']['hp']
        reward = 0.0

        # 打中 Dennis 最重要
        reward += (-e_hp_delta) * 2.0

        # 被打惩罚小
        reward += d_hp_delta * 0.1

        # 胜负
        if curr['dennis']['hp'] <= 0:
            reward += 200.0
        if curr['davis']['hp'] <= 0:
            reward -= 10.0

        # 距离奖励：越近越好
        dx = abs(curr['davis']['x'] - curr['dennis']['x'])
        reward += max(0.0, (300.0 - dx) / 300.0) * 0.5

        # 每帧小惩罚，鼓励尽快出手
        reward -= 0.01

        return reward

    def _wait_for_state(self, timeout=60.0):
        start = time.time()
        while True:
            with self._lock:
                if self.latest_state is not None:
                    return
            if time.time() - start > timeout:
                raise TimeoutError("等待游戏状态超时")
            time.sleep(0.01)

    def reset(self, seed=None, options=None):
        self._episode += 1
        print(f"[Reset] 第 {self._episode} 局开始，等待游戏连接...")
        # 清空状态，等待全新一局（time 必须从很小的值开始）
        with self._lock:
            self.latest_state = None
            self.prev_state = None
        # 等新状态
        self._wait_for_state()
        # 确保是新一局：time < 10 说明刚刚 reload
        start = time.time()
        while True:
            with self._lock:
                s = self.latest_state
            if s and s['time'] < 10:
                break
            if time.time() - start > 30:
                break
            with self._lock:
                self.latest_state = None
            self._wait_for_state()
        with self._lock:
            obs = self._state_to_obs(self.latest_state)
        print(f"[Reset] 已连接，开始第 {self._episode} 局")
        return obs, {}

    def step(self, action):
        with self._lock:
            self.prev_state = self.latest_state
            prev_time = self.latest_state['time'] if self.latest_state else -1

        self._send_action(action)

        # 等下一帧，最多等 3 秒
        deadline = time.time() + 3.0
        frame_received = False
        while True:
            with self._lock:
                curr = self.latest_state
            if curr and curr['time'] != prev_time:
                frame_received = True
                break
            if time.time() > deadline:
                break
            time.sleep(0.001)

        with self._lock:
            curr = self.latest_state

        # 帧超时：浏览器卡住了
        if not frame_received:
            print(
                f"[结果] 🔄 帧超时 | prev_time={prev_time} | curr_time={curr['time'] if curr else 'None'}")
            with self._lock:
                obs = self._state_to_obs(self.latest_state)
            return obs, -5.0, False, True, {}

        obs = self._state_to_obs(curr)
        reward = self._compute_reward(self.prev_state, curr)

        # 正常结束：有人死亡
        terminated = curr['davis']['hp'] <= 0 or curr['dennis']['hp'] <= 0

        if terminated:
            if curr['dennis']['hp'] <= 0:
                print(
                    f"[结果] 🏆 Davis 赢！davis hp={curr['davis']['hp']} dennis hp={curr['dennis']['hp']}")
            else:
                print(
                    f"[结果] 💀 Davis 输！davis hp={curr['davis']['hp']} dennis hp={curr['dennis']['hp']}")

        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass
