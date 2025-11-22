"""
auth:BoYvan Wu

date:2024/3/20
IMG_REG_Env.v1

date:2024/5/11
IMG_REG_Env.v2
"""
import sys
# sys.path.append('/home/james/TianshouImgReg/ImgRegEnv')
import io
import math
from collections import deque
import random
import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import spaces
from utils import transform_key_points, transform_key_points_no_inv, count_distance, affine_2x3_to_3x3, get_sift_features
from image_preprocess import move, preprocess_all_images, generate_affine_matrix_fixed, generate_random_affine_matrix, apply_affine_transform_cv2, apply_random_nonrigid_torch
import matplotlib.pyplot as plt
import torch
import pygame
import math
from api import request_sana_sample

class ImgRegEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, parallel, data_list, save_path, max_step, render_mode=None, env_mode="Easy", rand_cfg=None, re_randomize_each_reset=False):
        super(ImgRegEnv, self).__init__()

        self.action_space = spaces.Discrete(8)  # 动作空间
        self.render_mode = render_mode
        self.window_size = 400
        self.window = None
        self.clock = None

        self.rever_actions = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}  # 逆动作字典
        # Precompute 8 action affine matrices (CPU) to avoid per-step construction cost
        self.action_mats = [generate_affine_matrix_fixed(i, device='cpu') for i in range(8)]

        self.history = []  # 添加一个属性来存储动作历史
        self.reward_history = []  # 添加一个属性来存储动作历史对应的奖励
        self.one_time_rewards = {
            5: False,
            4.5: False,
            4: False,
            3.5: False,
            3: False
        }

        self.max_step = max_step

        self.observation_space = spaces.Box(  # 观测空间
            low=0.0, high=255.0,  # 当前 observation 未做 0-1 归一化，保持像素范围
            shape=(3, 256, 256),  # (C, H, W)
            dtype=np.float32
        )

        self.parallel = parallel  # 判断是否是多环境并行，如果是，数据加载方式会不一样
        self.save_path = save_path
        self.device = "cpu"  # 用于确定当前环境所处的设备

        # -------- 合成数据 API 设置 --------
        self.synth_prob = 0.9  # 每次 reset 走合成数据的概率
        # 其余网络请求/重试/提示词已挪到 api.py
        # ---------------------------------

        # -------- 非刚性微小形变设置（可选）--------
        # 为 floating/current_floating 叠加一个平滑小位移场（模拟切片褶皱/漂移）
        # 注意：一旦开启，图像关系不再是纯仿射，reward 仍按仿射矩阵距离会带一点噪声。
        self.nonrigid_prob = 1.0  # 默认关闭；>0 以概率启用
        self.nonrigid_cfg = {
            "max_disp": 3.0,      # 像素级形变幅度（建议 1-3）
            "smooth_sigma": 20.0, # 平滑程度（越大越平滑）
        }
        # ---------------------------------

        # 随机扰动配置（可热更新）与 reset 重随机开关
        self.rand_cfg = rand_cfg or {"scale_variation": 0.2, "rotation_variation": 30, "translation_variation": 60}
        self.re_randomize_each_reset = re_randomize_each_reset

        self.reference_image = torch.zeros(1, 256, 256)  # 基准图像
        self.floating_image = torch.zeros(1, 256, 256)  # 浮动图像
        self.ground_truth_image = torch.zeros(1, 256, 256)  # 地标图像
        self.current_floating_image = torch.zeros(1, 256, 256)  # 计算过程中的浮动图像

        # self.ground_truth_matrix = torch.eye(3)  # 基准矩阵
        self.current_matrix = torch.eye(3)  # 计算过程中的变换矩阵

        self.distance = torch.zeros(1)  # 当前环境中两张图像的距离
        self.round_num = 0

        self.kps = None
        self.gt_kps = None
        self.cu_kps = None
        self.kps_h = None  # homogeneous coords of self.kps, cached per episode

        self.frame = None

        self.ground_truth_matrix_inv = None

        if parallel:
            self.datalist = data_list  # 获取存储在cpu内存中的数据样本
        else:
            if env_mode == "Easy":
                he_folder = '/Users/wuboyuan/PycharmProjects/DRL_ImageRegistrationEnv/data/Cricle_img'
                cdx_folder = '/Users/wuboyuan/PycharmProjects/DRL_ImageRegistrationEnv/data/Cricle_img'
            else:
                he_folder = '/Users/wuboyuan/PycharmProjects/DRL_ImageRegistrationEnv/data/HE_image'
                cdx_folder = '/Users/wuboyuan/PycharmProjects/DRL_ImageRegistrationEnv/data/CDX_image'
            self.datalist = preprocess_all_images(he_folder_path=he_folder, cdx_folder_path=cdx_folder, rand_cfg=self.rand_cfg)
            print(f"[Env info]Date Length:{len(self.datalist)}")

    def set_rand_cfg(self, **kwargs):
        """热更新随机扰动范围。在训练过程中可随时调用。
        支持的键：scale_variation（比例），rotation_variation（度），translation_variation（像素）。
        仅更新传入的键；其余保持不变。
        """
        allowed = {"scale_variation", "rotation_variation", "translation_variation"}
        for k, v in kwargs.items():
            if k in allowed:
                self.rand_cfg[k] = float(v)

    def set_re_randomize_each_reset(self, flag: bool):
        """设置是否在每次 reset() 时根据 rand_cfg 重新随机初始扰动。"""
        self.re_randomize_each_reset = bool(flag)

    def re_randomize_now(self):
        """立即按当前 rand_cfg 生成新的初始扰动（热更新即时生效）。"""
        # 生成新的随机仿射 A（GT -> 新浮动）
        A_2x3 = generate_random_affine_matrix(**(self.rand_cfg or {}))
        A_3x3 = affine_2x3_to_3x3(A_2x3)
        self._report_perturbation_from_A(A_3x3, prefix='[re_randomize_now]')
        A_inv_3x3 = np.linalg.inv(A_3x3).astype(np.float32)  # (新浮动 -> GT)

        # 更新 GT 矩阵与逆
        self.ground_truth_matrix = torch.from_numpy(A_inv_3x3).float()
        self.ground_truth_matrix_inv = torch.from_numpy(A_3x3).float()

        # 用新的 A 在当前 GT 图像上生成新的浮动图
        self.floating_image = apply_affine_transform_cv2(self.ground_truth_image, A_2x3)

        # 刷新缓存与状态
        self.kps = transform_key_points_no_inv(self.gt_kps, self.ground_truth_matrix_inv)
        self.kps_h = np.hstack([
            self.kps.astype(np.float32),
            np.ones((self.kps.shape[0], 1), dtype=np.float32)
        ])
        self.current_floating_image = self.floating_image
        self.current_matrix = torch.eye(3)
        self.distance = self.get_distance()

    def _ensure_tensor_obs(self, obs):
        """Gym 返回的 obs 有时是 numpy.ndarray；这里统一转成 torch.Tensor。"""
        if isinstance(obs, np.ndarray):
            # 保持在 CPU；类型沿用 numpy 的 dtype（通常 float32）
            return torch.from_numpy(obs)
        return obs

    def _limit_kps(self, kps, max_n: int = 500):
        """Limit number of keypoints to avoid excessive compute / memory.
        Accepts None or (kps, desc) tuples; returns None if missing.
        """
        if kps is None:
            return None
        # Some upstream code may return (kps, desc) tuple.
        if isinstance(kps, tuple) or isinstance(kps, list):
            if len(kps) == 0:
                return None
            kps = kps[0]
            if kps is None:
                return None
        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim != 2:
            kps = kps.reshape(-1, 2)
        if kps.shape[1] != 2:
            # malformed, drop
            return None
        if kps.shape[0] > max_n:
            # Randomly subsample without replacement for diversity
            idx = np.random.choice(kps.shape[0], size=max_n, replace=False)
            kps = kps[idx]
        return kps

    def _request_synthetic_sample(self):
        """向 SANA API 请求一张合成图，并构造一个完整的数据样本：
        (reference_image, floating_image, ground_truth_image, ground_truth_matrix, gt_kps)

        约定：API 返回的合成图作为 ground_truth/reference；
        随机仿射 A 作用在 ground_truth 上生成 floating；
        ground_truth_matrix 为 floating -> ground_truth 的 3x3 矩阵；
        gt_kps 为 ground_truth 上的 SIFT 关键点。

        网络请求/重试/提示词选择放在 api.py。
        若请求失败则抛异常，由 reset() 兜底回退本地数据。
        """
        gt_np, prompt = request_sana_sample(
            height=256,
            width=256,
            # 如需改默认重试参数，可传 retry_wait/max_retries/api_url/prompts
        )

        # 1) 合成图作为 GT
        ground_truth_image = torch.from_numpy(gt_np).unsqueeze(0).float()  # (1,256,256)

        # 2) reference_image：这里没有跨模态信息，先与 GT 保持一致
        reference_image = ground_truth_image.clone()

        # 3) 生成随机仿射 A: GT -> floating
        A_2x3 = generate_random_affine_matrix(**(self.rand_cfg or {}))
        A_3x3 = affine_2x3_to_3x3(A_2x3)
        A_inv_3x3 = np.linalg.inv(A_3x3).astype(np.float32)  # floating -> GT
        ground_truth_matrix = torch.from_numpy(A_inv_3x3).float()

        # 4) 应用 A 得到 floating_image
        floating_image = apply_affine_transform_cv2(ground_truth_image, A_2x3)

        # 4.5) 可选：叠加微小非刚性形变场
        if self.nonrigid_prob > 0 and random.random() < self.nonrigid_prob:
            floating_image = apply_random_nonrigid_torch(
                floating_image,
                max_disp=self.nonrigid_cfg.get("max_disp", 2.0),
                smooth_sigma=self.nonrigid_cfg.get("smooth_sigma", 8.0),
                seed=None,
            )

        # 5) SIFT 关键点（在 GT 上提取）
        gt_kps = get_sift_features(gt_np)
        gt_kps = self._limit_kps(gt_kps, max_n=500)

        print(f"[Env info] Using synthetic sample from SANA. prompt={prompt[:60]}...")
        return reference_image, floating_image, ground_truth_image, ground_truth_matrix, gt_kps

    def _get_obs(self):
        """
        打包 observation：
        obs[0] -> ref_img
        obs[1] -> c_flt_img
        obs[2] -> 承载 [ref_kps, flo_kps] 的平面
        """
        # 1. 图像：2D (H, W)
        ref_img_np = self.reference_image.squeeze(0).byte().cpu().numpy().astype(np.uint8)
        c_flt_img_np = self.current_floating_image.squeeze(0).byte().cpu().numpy().astype(np.uint8)

        # 2. 关键点：做截断 + 补齐，保证 (50, 2)
        def normalize_kps(kps, target_n=50):
            if kps is None:
                # 没有的话直接全 0
                return np.zeros((target_n, 2), dtype=np.float32)
            kps = np.asarray(kps, dtype=np.float32)

            # 保守一点：如果维度不对，尝试 reshape 或直接丢弃到 0
            if kps.ndim != 2:
                kps = kps.reshape(-1, 2)
            if kps.shape[1] != 2:
                # 列数不对直接全 0，避免莫名其妙炸掉
                return np.zeros((target_n, 2), dtype=np.float32)

            # Clip to canvas range to satisfy observation_space bounds
            # (cu_kps may go negative or beyond 255 after affine.)
            kps = np.nan_to_num(kps, nan=0.0, posinf=255.0, neginf=0.0)
            kps[:, 0] = np.clip(kps[:, 0], 0.0, 255.0)
            kps[:, 1] = np.clip(kps[:, 1], 0.0, 255.0)

            n = kps.shape[0]
            if n >= target_n:
                return kps[:target_n]
            else:
                pad = np.zeros((target_n - n, 2), dtype=np.float32)
                return np.concatenate([kps, pad], axis=0)

        # ref_kps: 参考图像关键点（GT）
        # flo_kps: 当前浮动图像关键点（当前矩阵变换后的）
        ref_kps = normalize_kps(self.gt_kps, target_n=50)   # (50, 2)
        flo_kps = normalize_kps(self.cu_kps, target_n=50)   # (50, 2)

        # 这里你原来的 debug 可以保留 / 注释掉
        # print("gt_kps raw shape:", getattr(self.gt_kps, "shape", None))
        # print("cu_kps raw shape:", getattr(self.cu_kps, "shape", None))
        # print("ref_kps norm shape:", ref_kps.shape)
        # print("flo_kps norm shape:", flo_kps.shape)

        assert ref_kps.shape == (50, 2)
        assert flo_kps.shape == (50, 2)

        # 3. 打平成一个向量：[ref_kps, flo_kps]  -> (200,)
        kps_vec = np.concatenate(
            [ref_kps.reshape(-1), flo_kps.reshape(-1)],
            axis=0
        )  # 长度 200

        # 4. 准备一个和图像同样大小的平面来存 kps_vec
        kps_plane = np.zeros_like(ref_img_np, dtype=np.float32).flatten()
        assert kps_vec.size <= kps_plane.size, \
            f"kps 向量长度 {kps_vec.size} 超过了平面容量 {kps_plane.size}"

        kps_plane[:kps_vec.size] = kps_vec
        kps_plane = kps_plane.reshape(ref_img_np.shape)

        # 5. 最后 stack 成 (3, H, W)，和你的 unpack_observation 对齐
        obs = np.stack([ref_img_np, c_flt_img_np, kps_plane], axis=0).astype(np.float32)
        return obs


    def _get_info(self):
        """
        计算并返回基准矩阵和当前变换矩阵之间的Frobenius距离。
        """
        # 计算两个矩阵的差的Frobenius范数
        distance = torch.norm(self.ground_truth_matrix - self.current_matrix, p='fro')
        distance_numpy = np.array(distance.item())  # 将Python数值转换为NumPy数组

        # 返回包含距离的字典，现在距离是一个NumPy数组
        return {
            "distance": distance_numpy
        }

    def dataloader(self):
        """
        从内存中的数据中随机采样作为游戏环境
        :return:环境需要的数据
        """
        # 下面这部分代码要写在主函数里面，这样才能加载数据到内存中
        # root_folder='Expand_image/rotated_images'å
        # txt_file = 'Expand_image/new_affine_matrices.txt'
        # preprocess_all_images(root_folder=root_folder, txt_file_path=txt_file, target_size=(256, 256))
        random_int = random.randint(0, len(self.datalist) - 1)
        # print(random_int)
        data = self.datalist[random_int]
        reference_img, floating_img, ground_truth_img, ground_truth_matrix, kps = data
        return reference_img, floating_img, ground_truth_img, ground_truth_matrix, kps

    def _report_perturbation_from_A(self, A_3x3, prefix=""):
        """打印由 3x3 仿射矩阵推导出的扰动幅度（缩放、旋转、平移）。"""
        a, b, tx = float(A_3x3[0, 0]), float(A_3x3[0, 1]), float(A_3x3[0, 2])
        c, d, ty = float(A_3x3[1, 0]), float(A_3x3[1, 1]), float(A_3x3[1, 2])
        # 统一尺度（假设各向同性缩放）：s = sqrt(a^2 + c^2)
        s = math.sqrt(a * a + c * c)
        # 旋转角（度）：theta = atan2(c, a)
        rot_deg = math.degrees(math.atan2(c, a))
        print(f"[Env info]{prefix} rand_cfg={self.rand_cfg} | sampled => scale≈{s:.4f}, rot≈{rot_deg:.2f}°, tx={tx:.2f}, ty={ty:.2f}")

    def reset(self, seed=None, options=None):
        """
        用于初始化环境，开始下一局游戏
        :return:
        """
        super().reset(seed=seed)  # 不懂
        self.one_time_rewards = {
            5: False,
            4.5: False,
            4: False,
            3.5: False,
            3: False
        }
        self.history = []
        self.reward_history = []
        # --------- 概率性使用合成数据（API 生成） ---------
        if random.random() < self.synth_prob:
            try:
                (
                    self.reference_image,
                    self.floating_image,
                    self.ground_truth_image,
                    self.ground_truth_matrix,
                    self.gt_kps
                ) = self._request_synthetic_sample()
            except Exception as e:
                print(f"[Env warn] Synthetic sample failed, fallback to local dataloader: {e}")
                (
                    self.reference_image,
                    self.floating_image,
                    self.ground_truth_image,
                    self.ground_truth_matrix,
                    self.gt_kps
                ) = self.dataloader()
        else:
            (
                self.reference_image,
                self.floating_image,
                self.ground_truth_image,
                self.ground_truth_matrix,
                self.gt_kps
            ) = self.dataloader()
        # ------------------------------------------------
        # Limit keypoints count early to keep distance / kps_h stable
        self.gt_kps = self._limit_kps(self.gt_kps, max_n=500)
        self.ground_truth_matrix_inv = torch.linalg.inv(self.ground_truth_matrix)
        # 可选：每次 reset 时根据 rand_cfg 重新随机初始扰动
        if self.re_randomize_each_reset:
            A_2x3 = generate_random_affine_matrix(**(self.rand_cfg or {}))
            A_3x3 = affine_2x3_to_3x3(A_2x3)
            self._report_perturbation_from_A(A_3x3, prefix='[reset]')
            A_inv_3x3 = np.linalg.inv(A_3x3).astype(np.float32)
            self.ground_truth_matrix = torch.from_numpy(A_inv_3x3).float()
            self.ground_truth_matrix_inv = torch.from_numpy(A_3x3).float()
            self.floating_image = apply_affine_transform_cv2(self.ground_truth_image, A_2x3)
            # 可选：每次 reset 重新随机仿射后，也叠加一次微小非刚性形变
            if self.nonrigid_prob > 0 and random.random() < self.nonrigid_prob:
                self.floating_image = apply_random_nonrigid_torch(
                    self.floating_image,
                    max_disp=self.nonrigid_cfg.get("max_disp", 2.0),
                    smooth_sigma=self.nonrigid_cfg.get("smooth_sigma", 8.0),
                    seed=None,
                )
        # Precompute once per episode: transform gt_kps by GT inverse and cache homogeneous coordinates
        self.kps = transform_key_points_no_inv(self.gt_kps, self.ground_truth_matrix_inv)
        self.kps_h = np.hstack([
            self.kps.astype(np.float32),
            np.ones((self.kps.shape[0], 1), dtype=np.float32)
        ])
        self.round_num = 0
        self.current_floating_image = self.floating_image
        self.current_matrix = torch.eye(3)
        self.distance = self.get_distance()  # 获取初始距离
        if self.render_mode is not None:
            self._render_frame()
        observation = self._get_obs()
        info = self._get_info()
        # print(f"[Env info]:Env Reset")

        return observation, info

    # def _check_if_mostly_zeros(self, image, ratio=0.8):
    #     total_pixels = image.numel()
    #     zero_pixels = torch.sum(image == 0)
    #     return (zero_pixels / total_pixels) >= ratio

    def step(self, action):
        """
        调用一次代表智能体和环境做了一次交互
        :param action: 神经网络输出的代表动作的序号
        :return:当前state(基准图像+当前浮动图像), 奖励， 代表是否结束的变量
        """
        info = self._get_info()
        # 初始化一次性奖励标志
        # print(f"[Env info]{action}")

        if self.round_num <= self.max_step and torch.sum(self.current_floating_image) != 0:
            if self.distance <= 2.5:
                # 如果没到达结束回合但是上一轮游戏图像配准完成,或者初始图像已经很准了
                reward = 100
                print(f"[Env info]: Success!")
                self.reward_history.append(reward)
                observation = self._get_obs()
                return observation, reward, True, False, info
            else:  # 配准没有结束，继续配准
                (
                    self.current_floating_image,
                    self.current_matrix
                ) = move(
                    action_index=action,
                    image=self.floating_image,
                    matrix=self.current_matrix,
                    action_mats=self.action_mats
                )
                distance_0 = self.distance
                self.distance = self.get_distance()
                self.round_num += 1
                if self.render_mode is not None:
                    self._render_frame()
                reward_sign, sign, punish = self.enqueue_action(action)
                # print(f"penalty:{penalty}")
                # 计算奖励
                if reward_sign:
                    if (distance_0 - self.distance) <= 0:
                        base = 0
                    else:
                        # base = distance_0 - self.distance
                        base = 1  # 二值化奖励
                else:
                    base = 0

                reward = base + punish

                # 检查一次性奖励条件
                for threshold in self.one_time_rewards:
                    if self.distance <= threshold and not self.one_time_rewards[threshold]:
                        reward += 20  # 假设每个一次性奖励是50分,累加在原本的奖励上面
                        self.one_time_rewards[threshold] = True  # 标记为已领取
                        print(f"[Env info]: One-time reward of 50 for distance <= {threshold}!")

                observation = self._get_obs()
                if sign:
                    self.reward_history.append(reward)
                    self.history.append(action)

                # print("action list:")
                # for i in range(len(self.history)):
                #     print(self.history[i])

                # print("reward list:")
                # for i in range(len(self.reward_history)):
                #     print(self.reward_history[i])

                return (observation,
                        reward,
                        False, False, info)

        else:  # 到达结束回合的情况，或者图像全黑
            reward = -100
            self.reward_history.append(reward)
            observation = self._get_obs()
            return (observation,
                    reward,
                    True, False, info)

    def enqueue_action(self, action):
        # 检查动作是否合法
        if action not in self.rever_actions:
            print(f"Invalid action: {action}")
            return None

        # 查找逆动作
        inverse_action = self.rever_actions.get(action)

        punish = 0
        sign = True  # 决定当前动作和奖励是否入队
        reward_sign = True  # 决定当前奖励是非奖励给智能体
        if inverse_action in self.history:
            # 从后向前遍历，找到逆动作
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i] == inverse_action:
                    punish = -self.reward_history[i]
                    if punish == 0:
                        reward_sign = False
                    del self.history[i]
                    del self.reward_history[i]
                    sign = False
                    break  # 找到逆动作后不需要继续遍历
                else:
                    punish = 0

        return reward_sign, sign, punish

    def get_distance(self):
        """
        :return: 返回当前环境的距离度量(对数缩放)
        """
        # current_matrix -> numpy 2x3 (cheap conversion; 9 numbers)
        if isinstance(self.current_matrix, torch.Tensor):
            m = self.current_matrix.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            m = np.asarray(self.current_matrix, dtype=np.float32)
        m2x3 = m[:2, :]

        # Apply affine to cached homogeneous keypoints (N x 3) -> (N x 2)
        self.cu_kps = self.kps_h @ m2x3.T

        # Compute distance and log-scale it
        dist = count_distance(self.gt_kps, self.cu_kps)
        return math.log(dist + 1.0)

    def _render_frame(self):
        """
        Render and save the current state images as a single figure, including
        feature points detected by SIFT, but only those within the canvas.

        :return: img 1 frame
        """
        images = [self.reference_image, self.floating_image,
                  self.ground_truth_image, self.current_floating_image]
        titles = ['Reference Image', 'Floating Image',
                  'Ground Truth Image', 'Current Floating Image']

        # 获取self.floating_image的尺寸
        image_height, image_width = [256, 256]

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Adjusted to accommodate 4 subplots

        # Plot the original images
        for i, (image, title) in enumerate(zip(images, titles)):
            ax = axes[i // 2][i % 2]  # Get the current subplot
            ax.imshow((image.cpu().squeeze() / 255.0) if len(image.shape) > 2 else image / 255.0, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        mixed_ax = axes[0, 2]  # 选择(0, 2)位置的轴
        ref_img_np = self.reference_image.squeeze(0).cpu().numpy().astype(np.uint8)
        cu_flt_img_np = self.current_floating_image.squeeze(0).cpu().numpy().astype(np.uint8)
        # 使用alpha混合两个图像
        mixed_image = 0.5 * ref_img_np + 0.5 * cu_flt_img_np
        mixed_ax.imshow(mixed_image.astype(np.uint8), cmap='gray')
        mixed_ax.set_title('Mixed Image')
        mixed_ax.axis('off')

        # Plot the SIFT feature points only if they are within the canvas
        def plot_features(ax, features, title):
            # 过滤掉坐标超出图像边界的点
            valid_features = [(x, y) for x, y in features if 0 <= x < image_width and 0 <= y < image_height]
            # 绘制特征点
            ax.scatter([p[0] for p in valid_features], [image_height - p[1] for p in valid_features], color='red',
                       marker='x')  # 注意y坐标的处理
            ax.set_title(title)
            # 设置坐标轴的范围以适应画布大小，并反转y轴
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height - 1, 0)  # 注意y轴的取值范围颠倒了
            ax.invert_yaxis()  # 反转y轴
            ax.axis('off')

        # 假设self.kps, self.gt_kps, self.cu_kps是[x, y]坐标列表
        plot_features(axes[2, 0], self.kps, 'Detected Keypoints')
        plot_features(axes[2, 1], self.gt_kps, 'Ground Truth Keypoints')
        plot_features(axes[2, 2], self.cu_kps, 'Current Keypoints')

        distance_str = f'Distance: {self.distance:.4f}'
        for ax in axes.flatten():
            # 将文本放置在右上角，bbox 参数用于设置文本框的属性
            ax.text(1, 1, distance_str, transform=ax.transAxes, fontsize=12,
                    ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))

        buf = io.BytesIO()
        # 可以指定 DPI 来降低分辨率，例如 50
        fig.savefig(buf, format='png', dpi=50, bbox_inches='tight')
        buf.seek(0)

        # 使用PIL打开图像，降低颜色深度到256色
        image = Image.open(buf).convert('P')

        # 去除Alpha通道，只保留RGB通道
        image = image.convert('RGB')

        self.frame = np.array(image)

        # 将图像数据添加到队列
        if self.render_mode == "human":
            self.frame = np.flipud(self.frame)
            self.frame = np.rot90(self.frame, -1)

        # 关闭图形以释放内存
        plt.close(fig)

    def render(self):
        if self.render_mode is None:
            return None
        if self.frame is None:
            self._render_frame()
        # 调用内部的_render_frame方法来生成帧
        frame_array = self.frame
        # print(frame_array.shape)

        if self.render_mode == "human":
            # 确保 pygame 已经初始化，并且有一个窗口和时钟
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()

            # 将 NumPy 数组转换为 pygame Surface 对象
            frame_surface = pygame.surfarray.make_surface(frame_array)

            # 绘制到窗口
            self.window.fill((0, 0, 0))  # 清除屏幕
            self.window.blit(frame_surface, (0, 0))
            pygame.display.flip()  # 更新整个屏幕的Surface

            # 确保human渲染发生在预定义的帧率
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            # 如果是rgb_array模式，直接返回帧的NumPy数组
            return frame_array
        else:
            raise ValueError("Unsupported render mode: {}".format(self.render_mode))

    def close(self):
        """
        清理环境，释放资源。
        """
        # 释放任何创建的资源，如图像、数据结构等
        self.reference_image = None
        self.floating_image = None
        self.ground_truth_image = None
        self.current_floating_image = None
        # self.gif_list.clear()

        # 如果有打开的文件或网络连接，也应该在这里关闭
        # 示例：
        # if self.some_open_file:
        #     self.some_open_file.close()
        #     self.some_open_file = None

        # 打印一条消息，确认环境已被关闭
        print(f"[Env info]: Environment closed.")
