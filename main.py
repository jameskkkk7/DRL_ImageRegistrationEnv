import gymnasium as gym
from env import ImgRegEnv  # 确保从正确的模块导入

# 注册环境
gym.register(
    id='img_registration-v0',
    entry_point='env:ImgRegEnv',
    description='A Simple Image Registration Environment :)',
)
