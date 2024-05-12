import datetime
import os
import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def tensor2numpy(tensor):
    """
    将张量转换为 NumPy 数组。

    Args:
        tensor: 要转换的张量。

    Returns:
        NumPy 数组表示的张量。
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()


def numpy2tensor(array):
    """
    将 NumPy 数组转换为张量。

    Args:
        array: 要转换的 NumPy 数组。

    Returns:
        张量表示的 NumPy 数组。
    """
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array)


def affine_2x3_to_3x3(affine_2x3):
    """
    将2x3的仿射矩阵转换为3x3

    Args:
        affine_2x3: 输入的2x3仿射矩阵，可以是torch.tensor或numpy数组

    Returns:
        输出的3x3仿射矩阵，数据格式与输入相同
    """
    if isinstance(affine_2x3, torch.Tensor):
        affine_3x3 = torch.zeros((3, 3), dtype=affine_2x3.dtype, device=affine_2x3.device)
        # 将输入的2x3矩阵复制到新矩阵的左上角
        affine_3x3[:2, :3] = affine_2x3
        # 将[0, 0, 1]作为新矩阵的第三行
        affine_3x3[2, :] = torch.tensor([0, 0, 1], dtype=affine_2x3.dtype, device=affine_2x3.device)
        return affine_3x3
    elif isinstance(affine_2x3, np.ndarray):
        return np.vstack([affine_2x3, np.array([0, 0, 1], dtype=affine_2x3.dtype)])
    else:
        raise TypeError("输入数据类型不支持，只能是torch.Tensor或numpy数组")


def affine_3x3_to_2x3(affine_3x3):
    """
    将3x3的仿射矩阵转换为2x3

    Args:
        affine_3x3: 输入的3x3仿射矩阵，可以是torch.tensor或numpy数组

    Returns:
        输出的2x3仿射矩阵，数据格式与输入相同
    """

    if isinstance(affine_3x3, torch.Tensor):
        return affine_3x3[:2, :]
    elif isinstance(affine_3x3, np.ndarray):
        return affine_3x3[:2, :]
    else:
        raise TypeError("输入数据类型不支持，只能是torch.Tensor或numpy数组")


def tenser2index(tensor):
    """
    将张量转换为 One-hot 编码并返回最大值索引。

    Args:
        tensor: 要转换的张量。

    Returns:
        onehot_tensor: One-hot 编码表示的张量。
        max_index: 最大值索引。
    """
    # 确定张量的形状，以便在创建One-hot编码时保持相同的形状
    tensor = tensor.squeeze(0)
    shape = tensor.shape
    # print(tensor, shape)
    # 创建 One-hot 编码
    onehot_tensor = torch.zeros(shape, dtype=torch.long)

    # 使用 advanced indexing 为每个元素创建One-hot编码
    _, indices = torch.max(tensor, dim=0)
    onehot_tensor[indices] = 1

    # 返回 One-hot 编码的张量和最大值索引
    return indices.item()


def same_seeds(seed):
    """
    保证整个环境中的随机种子统一，保证实验的可复现能力

    :param seed: 你需要的种子
    :return: None
    """
    # 控制整个工程的随机种子相同，保证复现能力。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果CUDA可用，也为CUDA设置相同的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_experiment_folder(base_path='./result'):
    """
    根据时间生成一个记录当前实验数据的文件夹，返回其路径

    :param: base_path 基础路径
    :return: 生成的路径
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(base_path, current_time)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_experiment_config(save_path, config):
    """
    保存当前实验的参数

    :param save_path: 保存的路径
    :param config: 保存的内容
    :return: None
    """
    config_path = os.path.join(save_path, 'config.txt')
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f'{key} = {value}\n')


def get_sift_features(image):
    """
    Extract SIFT features from an image.

    Args:
        image: input image.

    Returns:
        tuple: (key_points, descriptors)
            - key_points: NumPy array of (x, y) coordinates for detected keypoints.
            - descriptors: NumPy array of SIFT descriptors for each keypoint.
    """

    # image = tensor2numpy(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    # print(image.shape, image.dtype)
    sift = cv2.SIFT_create()
    key_points = sift.detect(image, None)
    if key_points is None:
        print("No SIFT features detected.")
        return None, None
    key_points_np = np.array([(kp.pt[0], kp.pt[1]) for kp in key_points])
    # print(key_points_np, key_points_np.shape)
    return key_points_np


def transform_key_points(kps, m):
    """
    Apply an affine transformation to keypoints.

    Args:
        kps (np.array): Array of shape (n, 2) containing key_points.
        m (np.array): 2x3 affine transformation matrix.

    Returns:
        np.array: Transformed key_points.
    """
    transformed_kps = np.empty_like(kps)
    m = np.linalg.inv(m)
    if m.shape[0] == 3:
        m = affine_3x3_to_2x3(m)
    for i in range(kps.shape[0]):
        point = kps[i]
        point = np.hstack((point, [1.0]))
        transformed_point = m @ point.T
        transformed_kps[i] = transformed_point
    return transformed_kps


def count_distance(points1, points2):
    """
    Calculate the total Euclidean distance between two sets of points.

    Args:
        points1 (np.array): Array of shape (n1, 2) containing points in set 1.
        points2 (np.array): Array of shape (n2, 2) containing points in set 2.

    Returns:
        float: Total Euclidean distance between the two sets of points.
    """

    # 确保输入是NumPy数组
    points1 = np.array(points1)
    points2 = np.array(points2)

    # print(points1, points2)
    # 计算两堆点之间的欧氏距离
    distances = np.sum((points1 - points2) ** 2, axis=1)

    # 计算所有距离的平均值
    average_distance = np.mean(distances)

    return np.abs(average_distance)


def visualize_losses(train_losses, gpu_id, save_path):
    """
    Visualize training and validation losses and save the plot.

    Args:
    - gui_id: ID of GPU
    - train_losses: List of training losses.
    - save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'[GPU{gpu_id}]:Training Loss')
    plt.legend()
    plt.savefig(save_path+"/Loss.png")  # Save the figure
    plt.close()  # Close the plotting window to free up memory
