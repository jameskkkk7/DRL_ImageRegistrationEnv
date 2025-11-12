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
from utils import transform_key_points, transform_key_points_no_inv, count_distance
from image_preprocess import move, preprocess_all_images, generate_affine_matrix_fixed
import matplotlib.pyplot as plt
import torch
import pygame
import math


class ImgRegEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, parallel, data_list, save_path, max_step, render_mode=None, env_mode="Easy"):
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
            low=0.0, high=255.0,  # 归一化后的数据的值域为[0,1]
            shape=(2, 256, 256),  # 图像的高度、宽度和通道数
            dtype=np.uint8
        )

        self.parallel = parallel  # 判断是否是多环境并行，如果是，数据加载方式会不一样
        self.save_path = save_path
        self.device = "cpu"  # 用于确定当前环境所处的设备

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

            self.datalist = preprocess_all_images(he_folder_path=he_folder, cdx_folder_path=cdx_folder)
            print(f"[Env info]Date Length:{len(self.datalist)}")

    def _get_obs(self):
        """
        收集并返回环境的观测值。
        """
        # 直接转换为 np.uint8 类型，避免重复缩放
        gt_img_np = self.ground_truth_image.squeeze(0).cpu().numpy().astype(np.uint8)
        c_flt_img_np = self.current_floating_image.squeeze(0).cpu().numpy().astype(np.uint8)
        # 堆叠图像以形成 (2, 256, 256) 的数组
        obs = np.stack((gt_img_np, c_flt_img_np), axis=0)
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
        # root_folder='Expand_image/rotated_images'
        # txt_file = 'Expand_image/new_affine_matrices.txt'
        # preprocess_all_images(root_folder=root_folder, txt_file_path=txt_file, target_size=(256, 256))
        random_int = random.randint(0, len(self.datalist) - 1)
        # print(random_int)
        data = self.datalist[random_int]
        reference_img, floating_img, ground_truth_img, ground_truth_matrix, kps = data
        return reference_img, floating_img, ground_truth_img, ground_truth_matrix, kps

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
        (
            self.reference_image,
            self.floating_image,
            self.ground_truth_image,
            self.ground_truth_matrix,
            self.gt_kps
         ) = self.dataloader()
        self.ground_truth_matrix_inv = torch.linalg.inv(self.ground_truth_matrix)
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
