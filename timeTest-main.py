import multiprocessing as mp
import time
import torch

from env import ImgRegEnv

# ==== Helpers to enforce boundary & simulate Agent-side work ====
def _assert_cpu_tensor(x, where: str = "obs"):
    assert isinstance(x, torch.Tensor), f"{where} must be torch.Tensor, got {type(x)}"
    assert x.device.type == "cpu", f"{where} must be on CPU, got {x.device}"
    assert x.ndim >= 2, f"{where} should have at least 2 dims, got shape={tuple(getattr(x, 'shape', []))}"

def _agent_postprocess(obs: torch.Tensor, device: str = 'cpu', normalize: bool = False) -> torch.Tensor:
    """
    Simulate Agent-side processing ONLY (outside env): optional device move and normalization.
    This function must not be called inside the environment implementation.
    """
    y = obs
    if device != 'cpu' and torch.cuda.is_available():
        y = y.to(device, non_blocking=True)
    if normalize:
        y = y.float()
        y = y.div_(255.0)
    return y


# Helper function for multiprocessing parallel benchmark
def _run_env_benchmark(args):
    """
    Worker function for running a benchmark in a separate process.

    Args:
        args: tuple (env_id, data_list, max_env_steps, num_episodes)

    Returns:
        (total_steps, total_time) for this worker.
    """
    env_id, data_list, max_env_steps, num_episodes, agent_device, do_agent_post = args

    env = ImgRegEnv(
        parallel=True,
        data_list=data_list,
        save_path=f"result/mp_env_{env_id}",
        max_step=max_env_steps,
        render_mode=None,
        env_mode="Easy",
    )

    total_steps = 0
    total_step_time = 0.0  # pure env.step timing
    total_agent_time = 0.0  # simulated Agent-side time (device move/normalize)

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            _assert_cpu_tensor(obs, "reset_obs")
            terminated = False
            truncated = False
            step_in_ep = 0

            while not (terminated or truncated) and step_in_ep < max_env_steps:
                action = env.action_space.sample()

                t0 = time.time()
                obs, reward, terminated, truncated, info = env.step(action)
                t1 = time.time()
                _assert_cpu_tensor(obs, "step_obs")

                total_step_time += (t1 - t0)

                if do_agent_post:
                    t2 = time.time()
                    _ = _agent_postprocess(obs, device=agent_device, normalize=False)
                    t3 = time.time()
                    total_agent_time += (t3 - t2)

                total_steps += 1
                step_in_ep += 1
    finally:
        env.close()

    return total_steps, total_step_time, total_agent_time


def time_test(num_episodes: int = 5, max_env_steps: int = 100, *, agent_device: str = 'cpu', do_agent_post: bool = False):
    """
    使用随机 agent 测试环境响应时间（单进程单环境基准）。

    计时口径：
      - step_time: 仅 env.step(action) 的耗时（严格环境边界内）。
      - agent_time: 可选，模拟 Agent 侧（跨设备/归一化）的耗时（严格环境边界外）。
    """
    env = ImgRegEnv(
        parallel=False,
        data_list=None,
        save_path="result",
        max_step=max_env_steps,
        render_mode=None,
        env_mode="Easy",
    )

    step_times = []
    agent_times = []
    total_steps = 0

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            _assert_cpu_tensor(obs, "reset_obs")
            terminated = False
            truncated = False
            step_in_ep = 0

            while not (terminated or truncated) and step_in_ep < max_env_steps:
                action = env.action_space.sample()

                t0 = time.time()
                obs, reward, terminated, truncated, info = env.step(action)
                t1 = time.time()
                _assert_cpu_tensor(obs, "step_obs")
                step_times.append(t1 - t0)

                if do_agent_post:
                    t2 = time.time()
                    _ = _agent_postprocess(obs, device=agent_device, normalize=False)
                    t3 = time.time()
                    agent_times.append(t3 - t2)

                total_steps += 1
                step_in_ep += 1

            print(f"[TimeTest] Episode {ep + 1}: steps = {step_in_ep}")

    finally:
        env.close()

    if total_steps > 0:
        avg_step = sum(step_times) / total_steps if step_times else 0.0
        avg_agent = sum(agent_times) / len(agent_times) if agent_times else 0.0
        avg_total = avg_step + avg_agent
        print("================ Time Test Result ================")
        print(f"Episodes           : {num_episodes}")
        print(f"Total steps        : {total_steps}")
        print(f"Avg env.step       : {avg_step * 1000:.3f} ms/step")
        if do_agent_post:
            print(f"Avg agent post     : {avg_agent * 1000:.3f} ms/step  (device={agent_device})")
            print(f"Avg end-to-end     : {avg_total * 1000:.3f} ms/step")
            print(f"Throughput (E2E)   : {1.0 / avg_total:.2f} steps/sec")
        else:
            print(f"Throughput (step)  : {1.0 / avg_step:.2f} steps/sec")
        print("==================================================")
    else:
        print("[TimeTest] No steps were executed. Check environment or parameters.")

# Parallel benchmark function
def time_test_parallel(num_envs: int = 4,
                       num_episodes_per_env: int = 5,
                       max_env_steps: int = 100,
                       *,
                       agent_device: str = 'cpu',
                       do_agent_post: bool = False):
    """
    使用多进程并行多个 ImgRegEnv 实例，测试在清晰边界下的整体吞吐量。

    - 仅统计 env.step（环境边界内）时间；
    - 可选统计 Agent 侧（环境边界外）的耗时。
    """
    # 先在主进程构造一个环境，用于预处理数据并拿到 datalist，避免每个子进程重复预处理
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

    args_list = [
        (env_id, data_list, max_env_steps, num_episodes_per_env, agent_device, do_agent_post)
        for env_id in range(num_envs)
    ]

    with mp.Pool(processes=num_envs) as pool:
        results = pool.map(_run_env_benchmark, args_list)

    total_steps = sum(r[0] for r in results)
    total_step_time = sum(r[1] for r in results)
    total_agent_time = sum(r[2] for r in results)

    if total_steps > 0 and total_step_time > 0.0:
        avg_step = total_step_time / total_steps
        avg_agent = (total_agent_time / total_steps) if do_agent_post and total_agent_time > 0 else 0.0
        avg_total = avg_step + avg_agent
        print("============ Parallel Time Test Result ============")
        print(f"Num envs             : {num_envs}")
        print(f"Episodes/env         : {num_episodes_per_env}")
        print(f"Total steps (all)    : {total_steps}")
        print(f"Avg env.step (all)   : {avg_step * 1000:.3f} ms/step")
        if do_agent_post:
            print(f"Avg agent post (all) : {avg_agent * 1000:.3f} ms/step  (device={agent_device})")
            print(f"Avg end-to-end (all) : {avg_total * 1000:.3f} ms/step")
            print(f"Throughput (E2E all) : {1.0 / avg_total:.2f} steps/sec")
        else:
            print(f"Throughput (step)    : {total_steps / total_step_time:.2f} steps/sec")
        print("===================================================")
    else:
        print("[Parallel TimeTest] No steps were executed. Check environment or parameters.")


if __name__ == "__main__":
    # 自动选择 Agent 设备（仅在 Agent 侧使用，环境始终在 CPU 上输出 Tensor）
    # agent_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent_device = 'cpu'


    # 单进程单环境基准测试（按需调整 episode/steps）
    time_test(num_episodes=100, max_env_steps=500, agent_device=agent_device, do_agent_post=False)

    # 多进程并行基准测试
    time_test_parallel(num_envs=3, num_episodes_per_env=100, max_env_steps=500,
                       agent_device=agent_device, do_agent_post=False)