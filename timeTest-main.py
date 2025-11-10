import time

from env import ImgRegEnv


def time_test(num_episodes: int = 5, max_env_steps: int = 100):
    """
    使用随机 agent 测试环境响应时间。

    每一步的耗时统计区间：
        t0 = 调用 env.step(action) 之前
        t1 = 调用 env.render() 之后
    即一次完整交互 = step + render。

    :param num_episodes: 运行多少个 episode
    :param max_env_steps: 每个 episode 最多允许的步数（也会传给环境的 max_step）
    """

    # 注意：render_mode=None 是有意为之，这样可以避免在基准测试时调用 _render_frame() 带来的额外开销，
    # 专注于纯 env.step 性能。如果你需要测试渲染性能，请将 render_mode 改为 "rgb_array" 或 "human"。
    env = ImgRegEnv(
        parallel=False,
        data_list=None,
        save_path="result",
        max_step=max_env_steps,
        render_mode=None,
        env_mode="Easy",  # 你的数据路径如果用 HE/CDX 那套，就改成 "Hard"
    )

    step_times = []
    total_steps = 0

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            step_in_ep = 0

            while not (terminated or truncated) and step_in_ep < max_env_steps:
                # 随机 agent：从动作空间中随机采样一个动作
                action = env.action_space.sample()

                t0 = time.time()
                obs, reward, terminated, truncated, info = env.step(action)

                # 如果你只想测 step 的耗时，把下面这行注释掉即可
                # env.render()

                t1 = time.time()

                step_times.append(t1 - t0)
                total_steps += 1
                step_in_ep += 1

            print(f"[TimeTest] Episode {ep + 1}: steps = {step_in_ep}")

    finally:
        env.close()

    if total_steps > 0:
        avg_time = sum(step_times) / total_steps
        print("================ Time Test Result ================")
        print(f"Episodes      : {num_episodes}")
        print(f"Total steps   : {total_steps}")
        print(f"Avg step time : {avg_time * 1000:.3f} ms/step")
        print(f"Throughput    : {1.0 / avg_time:.2f} steps/sec")
        print("==================================================")
    else:
        print("[TimeTest] No steps were executed. Check environment or parameters.")


if __name__ == "__main__":
    # 这里可以根据需要随便改参数
    time_test(num_episodes=5, max_env_steps=100)