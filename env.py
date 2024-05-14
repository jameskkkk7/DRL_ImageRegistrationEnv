"""
auth:BoYvan Wu

date:2024/3/20
IMG_REG_Env.v1

date:2024/5/11
IMG_REG_Env.v2
TODO: 更改一些图像处理的矩阵的存储位置，分配好内存和显卡，从而实现运行速度的加速
"""
import io
import math
import os
import random
import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import spaces
from utils import tenser2index, transform_key_points, count_distance
from image_preprocess import move, preprocess_all_images
import datetime
import matplotlib.pyplot as plt
import torch
import imageio
import pygame


class ImgRegEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, parallel, data_list, save_path, render_mode='human'):
        super(ImgRegEnv, self).__init__()

        self.action_space = spaces.Discrete(8)  # 动作空间
        self.render_mode = render_mode
        self.window_size = 256
        self.window = None
        self.clock = None

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

        self.ground_truth_matrix = torch.eye(3)  # 基准矩阵
        self.current_matrix = torch.eye(3)  # 计算过程中的变换矩阵

        self.distance = torch.zeros(1)  # 当前环境中两张图像的距离
        self.round_num = 0

        self.kps = None
        self.gt_kps = None
        self.cu_kps = None

        self.frame = None

        if parallel:
            self.datalist = data_list  # 获取存储在cpu内存中的数据样本
        else:
            root_folder = './Expand_image/rotated_images'
            txt_file = './Expand_image/new_affine_matrices.txt'
            self.datalist = preprocess_all_images(root_folder=root_folder, txt_file_path=txt_file)
            print(f"[Env info]Date Length:{len(self.datalist)}")

    def _get_obs(self):
        """
        收集并返回环境的观测值。
        """
        # 将图像数据乘以 255 并转换为 np.uint8 类型，去掉单一的维度，并堆叠起来
        gt_img_np = (self.ground_truth_image * 255).squeeze(0).byte().numpy().astype(np.uint8)
        c_flt_img_np = (self.current_floating_image * 255).squeeze(0).byte().numpy().astype(np.uint8)

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
        random_int = random.randint(0, len(self.datalist)-1)
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
        (
            self.reference_image,
            self.floating_image,
            self.ground_truth_image,
            self.ground_truth_matrix,
            self.kps
         ) = self.dataloader()
        self.round_num = 0
        self.current_floating_image = self.floating_image
        self.current_matrix = torch.eye(3)
        self.distance = self.get_distance()  # 获取初始距离
        observation = self._get_obs()
        info = self._get_info()
        # print(f"[Env info]:Env Reset")

        return observation, info

    def step(self, action):
        """
        调用一次代表智能体和环境做了一次交互
        :param action: 神经网络输出的代表动作的序号
        :return:当前state（基准图像+当前浮动图像）， 奖励， 代表是否结束的变量
        """
        info = self._get_info()
        if self.round_num <= 10 and torch.sum(self.current_floating_image) != 0:
            if self.distance <= 1:
                # 如果没到达结束回合但是上一轮游戏图像配准完成,或者初始图像已经很准了
                reward = 10
                print(f"[Env info]: Success!")
                observation = self._get_obs()
                return observation, reward, True, False, info
            else:  # 配准没有结束，继续配准
                (
                    self.current_floating_image,
                    self.current_matrix
                ) = move(
                    action_index=action,
                    image=self.floating_image,
                    matrix=self.current_matrix
                )
                distance_0 = self.distance
                self.distance = self.get_distance()
                self.round_num += 1
                self._render_frame()
                # 这里的奖励使用上一个时刻的距离和这一个时刻距离的差值，如果距离变小了那就是正数(奖励)，距离变大了就是负数（惩罚）
                observation = self._get_obs()
                return observation, distance_0 - self.distance, False, False, info
        else:  # 到达结束回合的情况，或者图像全黑
            reward = -10
            observation = self._get_obs()
            return observation, reward, True, False, info

    def get_distance(self):
        """

        :return: 返回当前环境的奖励
        """
        gt_m = torch.linalg.inv(self.ground_truth_matrix)
        self.gt_kps = transform_key_points(self.kps, gt_m)  # 因为一些奇怪的bug
        self.cu_kps = transform_key_points(self.kps, self.current_matrix)
        reward = count_distance(self.gt_kps, self.cu_kps)
        reward = math.log(reward + 1)
        return reward

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

        # Plot the SIFT feature points only if they are within the canvas
        def plot_features(ax, features, title):
            # 过滤掉坐标超出图像边界的点
            valid_features = [(x, y) for x, y in features if 0 <= x < image_width and 0 <= y < image_height]
            # 绘制特征点
            ax.scatter([p[0] for p in valid_features], [image_height - p[1] for p in valid_features], color='red', marker='x')  # 注意y坐标的处理
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

        buf = io.BytesIO()
        # 可以指定 DPI 来降低分辨率，例如 50
        fig.savefig(buf, format='png', dpi=50, bbox_inches='tight')
        buf.seek(0)

        # 使用PIL打开图像，降低颜色深度到256色
        image = Image.open(buf).convert('P')

        # 去除Alpha通道，只保留RGB通道
        image = image.convert('RGB')

        # 将图像数据添加到队列
        self.frame = np.array(image)

        # 关闭图形以释放内存
        plt.close(fig)

    # def render(self):
    #     """
    #     将self.gif_list中的所有帧渲染成一个GIF，并清空self.gif_list。
    #     """
    #     # 获取当前时间戳
    #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
    #     # 创建一个包含时间戳的文件名
    #     gif_filename = f'environment_animation_{timestamp}.gif'
    #     gif_path = os.path.join(self.save_path, gif_filename)

    #     # 使用 imageio 库保存 GIF
    #     imageio.mimsave(gif_path, self.gif_list, duration=0.1, fps=15, subrectangles=True)

    #     # 清空 gif_list 以便存储新的帧
    #     self.gif_list.clear()

    #     # 打印保存成功的消息
    #     print(f"Compressed GIF saved to {gif_path}")
    def render(self):
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


# def test_env():
#     # 假设你的数据列表已经准备好了，这里用一个空列表作为示例
#     root_folder = './Expand_image/rotated_images'
#     txt_file = './Expand_image/new_affine_matrices.txt'
#     data_list = preprocess_all_images(root_folder=root_folder, txt_file_path=txt_file, target_size=(256, 256))
#
#     # 初始化环境
#     env = ImgRegEnv(data_list=data_list, parallel=False, save_path="result", render_mode="human")
#
#     # 重置环境
#     env.reset()
#     # env.render()
#     # 进行一系列步骤，直到环境结束
#     terminated = False
#     step = 0
#     while not terminated:
#         # 这里假设action_tenser是一个从神经网络获取的动作张量，为了测试，我们随机生成一个动作
#         # 在实际应用中，你需要根据你的神经网络模型来生成这个动作
#         action = np.random.randint(8)
#
#         # 进行一步
#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         # print(observation)
#         # print("Shape:", observation.shape)
#
#         # 更新步骤计数
#         step += 1
#
#         # 打印奖励信息
#         print(f"Step {step}: Reward = {reward}")
#
#     # 渲染当前状态
#     # env.render()
#     print("Environment has ended.")
#
#
# # 运行测试函数
# test_env()

