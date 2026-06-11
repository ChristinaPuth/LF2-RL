from env import LF2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

print("初始化环境...")
env = LF2Env()

print("请用脚本启动浏览器，或手动打开：")
print("http://localhost:8000/F.LF/game/game.html?debug&headless")
print("等待连接...")

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,        # 原来 512，调大让每次更新见到更多样本
    batch_size=64,
    learning_rate=1e-4,
    n_epochs=5,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/",
)

print("开始训练！浏览器最小化即可，训练在后台继续")
model.learn(total_timesteps=500_000)   # 原来 100k，至少跑 500k 才能看到学习效果
model.save("lf2_ppo")
print("训练完成！模型已保存为 lf2_ppo.zip")
