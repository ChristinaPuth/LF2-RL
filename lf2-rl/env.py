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

class LF2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)
        self.ws = None
        self.latest_state = None
        self.prev_state = None
        self._lock = threading.Lock()
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
        return np.array([
            d['x'], d['y'], d['hp'], d['mp'],
            float(d['frame']), dir_map.get(d['dir'], 0),
            e['x'], e['y'], e['hp'], e['mp'],
            float(e['frame']), dir_map.get(e['dir'], 0),
        ], dtype=np.float32)

    def _compute_reward(self, prev, curr):
        if prev is None:
            return 0.0
        d_hp_delta = curr['davis']['hp'] - prev['davis']['hp']
        e_hp_delta = curr['dennis']['hp'] - prev['dennis']['hp']
        reward = 0.0
        reward += (-e_hp_delta) * 0.5
        reward += d_hp_delta * 0.1
        if curr['dennis']['hp'] <= 0:
            reward += 100.0
        if curr['davis']['hp'] <= 0:
            reward -= 10.0
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
        with self._lock:
            self.latest_state = None
            self.prev_state = None
        print("等待新一局（请刷新浏览器）...")
        self._wait_for_state()
        with self._lock:
            obs = self._state_to_obs(self.latest_state)
        return obs, {}

    def step(self, action):
        with self._lock:
            self.prev_state = self.latest_state
            prev_time = self.latest_state['time'] if self.latest_state else -1

        self._send_action(action)

        # 等下一帧
        deadline = time.time() + 1.0
        while True:
            with self._lock:
                curr = self.latest_state
            if curr and curr['time'] != prev_time:
                break
            if time.time() > deadline:
                break
            time.sleep(0.001)

        with self._lock:
            curr = self.latest_state

        obs = self._state_to_obs(curr)
        reward = self._compute_reward(self.prev_state, curr)
        terminated = curr['davis']['hp'] <= 0 or curr['dennis']['hp'] <= 0
        return obs, reward, terminated, False, {}

    def close(self):
        pass
