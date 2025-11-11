import multiprocessing as mp
import time

from env import ImgRegEnv


# Helper function for multiprocessing parallel benchmark
def _run_env_benchmark(args):
    """
    Worker function for running a benchmark in a separate process.

    Args:
        args: tuple (env_id, data_list, max_env_steps, num_episodes)

    Returns:
        (total_steps, total_time) for this worker.
    """
    env_id, data_list, max_env_steps, num_episodes = args

    # 每个子进程各自创建一个环境实例，使用 parallel=True 并复用主进程准备好的 data_list
    env = ImgRegEnv(
        parallel=True,
        data_list=data_list,
        save_path=f"result/mp_env_{env_id}",
        max_step=max_env_steps,
        render_mode=None,
        env_mode="Easy",
    )

    total_steps = 0
    total_time = 0.0

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            step_in_ep = 0

            while not (terminated or truncated) and step_in_ep < max_env_steps:
                action = env.action_space.sample()
                t0 = time.time()
                obs, reward, terminated, truncated, info = env.step(action)
                t1 = time.time()

                total_time += (t1 - t0)
                total_steps += 1
                step_in_ep += 1
    finally:
        env.close()

    return total_steps, total_time


def time_test(num_episodes: int = 5, max_env_steps: int = 100):
    """
    使用随机 agent 测试环境响应时间（单进程单环境基准）。

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

# Parallel benchmark function
def time_test_parallel(num_envs: int = 4,
                       num_episodes_per_env: int = 5,
                       max_env_steps: int = 100):
    """
    使用多进程并行多个 ImgRegEnv 实例，测试并行采样时的整体吞吐量。

    :param num_envs: 同时并行的环境数量（也是子进程数）
    :param num_episodes_per_env: 每个环境各自运行多少个 episode
    :param max_env_steps: 每个 episode 最多步数（传给环境的 max_step）
    """

    # 先在主进程构造一个环境，用于预处理数据并拿到 datalist，
    # 避免每个子进程都重复跑 preprocess_all_images。
    loader_env = ImgRegEnv(
        parallel=False,
        data_list=None,
        save_path="result",
        max_step=max_env_steps,
        render_mode=None,
        env_mode="Easy",
    )
    data_list = loader_env.datalist
    loader_env.close()

    # 为每个子进程准备参数
    args_list = [
        (env_id, data_list, max_env_steps, num_episodes_per_env)
        for env_id in range(num_envs)
    ]

    with mp.Pool(processes=num_envs) as pool:
        results = pool.map(_run_env_benchmark, args_list)

    total_steps = sum(r[0] for r in results)
    total_time = sum(r[1] for r in results)

    if total_steps > 0 and total_time > 0.0:
        avg_time = total_time / total_steps
        print("============ Parallel Time Test Result ============")
        print(f"Num envs            : {num_envs}")
        print(f"Episodes/env        : {num_episodes_per_env}")
        print(f"Total steps (all)   : {total_steps}")
        print(f"Avg step time (all) : {avg_time * 1000:.3f} ms/step")
        print(f"Throughput (all)    : {total_steps / total_time:.2f} steps/sec")
        print("===================================================")
    else:
        print("[Parallel TimeTest] No steps were executed. Check environment or parameters.")


if __name__ == "__main__":
    # 单进程单环境基准测试
    time_test(num_episodes=100, max_env_steps=500)

    # 多进程并行基准测试（可按需调整或注释）
    time_test_parallel(num_envs=3, num_episodes_per_env=100, max_env_steps=500)