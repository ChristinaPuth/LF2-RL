"""
train.py — Davis vs Dennis DQN 訓練主程式
使用 Double DQN + Replay Buffer + ε-greedy

放置路徑：C:\\Users\\n1003\\Desktop\\F\\train.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from rl_env import LF2Env

# =====================================================
# 超參數
# =====================================================
EPISODES        = 5000    # 總訓練場數
OBS_DIM         = 12      # 觀察值維度
ACT_DIM         = 11      # 動作數量

GAMMA           = 0.99    # 折扣因子
LR              = 1e-3    # 學習率
BATCH_SIZE      = 64      # 每次更新抽幾筆
BUFFER_SIZE     = 100_000 # Replay buffer 大小
WARMUP_STEPS    = 1000    # 多少步後才開始訓練
TARGET_UPDATE   = 1000    # 每幾步同步 target network

EPSILON_START   = 1.0     # 初始探索率
EPSILON_END     = 0.05    # 最低探索率
EPSILON_DECAY   = 50_000  # 多少步內從 start 降到 end

EVAL_EVERY      = 100     # 每幾場評估一次
EVAL_EPISODES   = 20      # 評估時跑幾場
SAVE_DIR        = 'checkpoints'

# =====================================================
# Q 網路（小 MLP，適合這個問題規模）
# =====================================================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# Replay Buffer
# =====================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_next)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buf)


# =====================================================
# ε-greedy 動作選擇
# =====================================================
def select_action(q_net, obs, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, ACT_DIM - 1)
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        q_vals = q_net(obs_t)
        return q_vals.argmax().item()


def get_epsilon(step):
    """線性衰減 epsilon"""
    ratio = min(1.0, step / EPSILON_DECAY)
    return EPSILON_START + ratio * (EPSILON_END - EPSILON_START)


# =====================================================
# Double DQN 更新
# =====================================================
def update(q_net, target_net, optimizer, buffer, device):
    if len(buffer) < BATCH_SIZE:
        return 0.0

    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)
    s      = s.to(device)
    a      = a.to(device)
    r      = r.to(device)
    s_next = s_next.to(device)
    done   = done.to(device)

    # 目前 Q 值
    q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Double DQN target：用 q_net 選動作，用 target_net 估值
    with torch.no_grad():
        next_actions = q_net(s_next).argmax(1)
        next_q = target_net(s_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = r + GAMMA * next_q * (1 - done)

    loss = nn.MSELoss()(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10)  # 梯度裁剪，防爆炸
    optimizer.step()

    return loss.item()


# =====================================================
# 評估（關掉 ε-greedy，純用 Q 值選動作）
# =====================================================
def evaluate(q_net, env, device, n_episodes=EVAL_EPISODES):
    wins = 0
    total_reward = 0.0

    q_net.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            ep_reward = 0
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = q_net(obs_t).argmax().item()
                obs, reward, done, info = env.step(action)
                ep_reward += reward
            total_reward += ep_reward
            if info == 'win':
                wins += 1
    q_net.train()

    win_rate   = wins / n_episodes
    avg_reward = total_reward / n_episodes
    return win_rate, avg_reward


# =====================================================
# 主訓練迴圈
# =====================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置：{device}')

    # 建立環境（會等瀏覽器連接）
    env = LF2Env()

    # 建立網路
    q_net      = QNetwork(OBS_DIM, ACT_DIM).to(device)
    target_net = QNetwork(OBS_DIM, ACT_DIM).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 訓練紀錄
    total_steps   = 0
    best_win_rate = 0.0
    history = []   # (episode, win_rate, avg_reward)

    print(f'\n開始訓練！共 {EPISODES} 場')
    print('=' * 60)

    for episode in range(1, EPISODES + 1):
        obs  = env.reset()
        done = False
        ep_reward   = 0.0
        ep_steps    = 0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0

        while not done:
            epsilon = get_epsilon(total_steps)
            action  = select_action(q_net, obs, epsilon, device)

            next_obs, reward, done, info = env.step(action)

            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs

            ep_reward += reward
            ep_steps  += 1
            total_steps += 1

            # 開始訓練（warmup 結束後）
            if total_steps >= WARMUP_STEPS:
                loss = update(q_net, target_net, optimizer, buffer, device)
                ep_loss_sum += loss
                ep_loss_cnt += 1

            # 同步 target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        # 每場結束：印簡單訊息
        result_str = {'win': '🏆 贏', 'lose': '💀 輸', 'timeout': '⏱ 超時'}.get(info, info)
        avg_loss   = ep_loss_sum / ep_loss_cnt if ep_loss_cnt > 0 else 0
        epsilon    = get_epsilon(total_steps)

        print(f'[Ep {episode:4d}] {result_str} | '
              f'reward={ep_reward:6.2f} | steps={ep_steps:4d} | '
              f'ε={epsilon:.3f} | loss={avg_loss:.4f}')

        # 每 EVAL_EVERY 場評估一次
        if episode % EVAL_EVERY == 0:
            print(f'\n--- 第 {episode} 場評估（{EVAL_EPISODES} 場）---')
            win_rate, avg_reward = evaluate(q_net, env, device)
            history.append((episode, win_rate, avg_reward))

            print(f'  勝率：{win_rate:.1%}  |  平均 reward：{avg_reward:.3f}')

            # 儲存最佳模型
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                path = os.path.join(SAVE_DIR, 'best_model.pt')
                torch.save(q_net.state_dict(), path)
                print(f'  ★ 新最佳勝率！模型已儲存到 {path}')

            # 每 500 場存一個 checkpoint
            if episode % 500 == 0:
                path = os.path.join(SAVE_DIR, f'ep{episode}.pt')
                torch.save(q_net.state_dict(), path)
                print(f'  Checkpoint 已儲存：{path}')

            print(f'-------------------------------------------\n')

    print('\n訓練完成！')
    print(f'最佳勝率：{best_win_rate:.1%}')

    # 儲存最終模型
    torch.save(q_net.state_dict(), os.path.join(SAVE_DIR, 'final_model.pt'))

    env.close()
    return history


if __name__ == '__main__':
    train()
