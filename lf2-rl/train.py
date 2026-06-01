from env import LF2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time

print("初始化环境...")
env = LF2Env()

print("请在浏览器打开游戏: http://localhost:8000/F.LF/game/game.html?debug")
print("等待连接...")

model = PPO(
    "MlpPolicy",
    env,
    n_steps=512,
    batch_size=64,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log="./logs/",
)

print("开始训练！每局结束后请刷新浏览器继续下一局")
model.learn(total_timesteps=100_000)
model.save("lf2_ppo")
print("训练完成！模型已保存")
