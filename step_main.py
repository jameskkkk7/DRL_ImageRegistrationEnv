import gymnasium as gym
from env import ImgRegEnv  # 确保从正确的模块导入
import time

def manual_test():
    """手动输入动作测试环境：输入 0-7 执行动作，r 重置，q 退出。"""
    env = gym.make(
        'img_registration-v0',
        parallel=False,
        data_list=None,
        save_path='result',
        max_step=200,
        render_mode='human',
        env_mode='Easy'
    )
    obs, info = env.reset()
    print("环境已重置。输入 0-7 选择动作；输入 r 重置；输入 q 退出。")
    step = 0
    try:
        while True:
            # 渲染一帧，便于可视化
            env.render()
            cmd = input("动作(0-7) / r 重置 / q 退出: ").strip().lower()
            if cmd == 'q':
                break
            if cmd == 'r':
                obs, info = env.reset()
                step = 0
                continue
            if not cmd.isdigit() or not (0 <= int(cmd) <= 7):
                print("非法输入，请输入 0-7、r 或 q。")
                continue
            action = int(cmd)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            print(f"Step {step}: action={action}, reward={reward:.3f}, distance={info.get('distance'):.4f}")
            if terminated or truncated:
                print("回合结束，自动重置。")
                obs, info = env.reset()
                step = 0
            # 小憩一下，避免终端刷屏过快
            time.sleep(0.02)
    finally:
        env.close()

# 注册环境
gym.register(
    id='img_registration-v0',
    entry_point='env:ImgRegEnv',
    # description='A Simple Image Registration Environment :)',
)


if __name__ == '__main__':
    manual_test()