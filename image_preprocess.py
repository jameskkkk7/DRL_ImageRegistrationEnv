import math
import os
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def apply_random_nonrigid_torch(
    image: torch.Tensor,
    max_disp: float = 2.0,
    smooth_sigma: float = 8.0,
    seed: int = None,
    padding_mode: str = "border",
):
    """
    对 (1,H,W) 的 CPU tensor 施加一个平滑的小非刚性形变场。

    image: torch.Tensor (1,H,W), CPU, float32/uint8
    max_disp: 位移幅度（像素级，建议 1-3）
    smooth_sigma: 高斯平滑 sigma，越大越“柔”
    """
    assert image.ndim == 3 and image.shape[0] == 1, f"expected (1,H,W), got {tuple(image.shape)}"
    H, W = int(image.shape[1]), int(image.shape[2])

    # RNG
    if seed is not None:
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))
        noise_x = torch.randn((H, W), generator=g, device='cpu')
        noise_y = torch.randn((H, W), generator=g, device='cpu')
    else:
        noise_x = torch.randn((H, W), device='cpu')
        noise_y = torch.randn((H, W), device='cpu')

    # 用 cv2 高斯模糊做平滑（避免引入 scipy）
    nx = noise_x.numpy().astype(np.float32)
    ny = noise_y.numpy().astype(np.float32)
    k = int(2 * round(smooth_sigma * 3) + 1)  # ~6 sigma window
    k = max(k, 3)
    if k % 2 == 0:
        k += 1
    nx = cv2.GaussianBlur(nx, (k, k), smooth_sigma)
    ny = cv2.GaussianBlur(ny, (k, k), smooth_sigma)

    # 归一化 + 缩放到像素位移
    nx = nx / (np.std(nx) + 1e-6) * max_disp
    ny = ny / (np.std(ny) + 1e-6) * max_disp

    disp_x = torch.from_numpy(nx)
    disp_y = torch.from_numpy(ny)

    # base grid in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device='cpu'),
        torch.linspace(-1.0, 1.0, W, device='cpu'),
        indexing='ij'
    )

    # 像素位移 -> 归一化位移
    disp_x_norm = disp_x * (2.0 / max(W - 1, 1))
    disp_y_norm = disp_y * (2.0 / max(H - 1, 1))

    grid = torch.stack([xx + disp_x_norm, yy + disp_y_norm], dim=-1)  # (H,W,2)

    # sample
    orig_dtype = image.dtype
    img = image.to(dtype=torch.float32, device='cpu').unsqueeze(0)  # (1,1,H,W)
    out = F.grid_sample(
        img,
        grid.unsqueeze(0),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)  # (1,H,W)

    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        out = out.clamp(0.0, 255.0).round().to(dtype=torch.uint8)
    return out

def apply_affine_transform_torch(image, affine_matrix, output_size=(256, 256)):
    """
    Apply an affine transformation to an image using pure PyTorch on CPU.

    Args:
        image (torch.Tensor): shape (1, H, W), dtype float32/uint8, CPU tensor.
        affine_matrix (torch.Tensor or np.ndarray): 3x3 or 2x3 affine matrix in *pixel* coordinates
            that maps OUTPUT pixel coordinates to INPUT pixel coordinates (CV2-style).
        output_size (tuple): (H, W) of the output image.

    Returns:
        torch.Tensor: Transformed image tensor with shape (1, H, W) and same dtype as input image.
    """
    if isinstance(affine_matrix, np.ndarray):
        affine_matrix = torch.from_numpy(affine_matrix)
    affine_matrix = affine_matrix.to(dtype=torch.float32, device='cpu')
    if affine_matrix.shape[0] == 2:
        # make it 3x3
        affine_matrix = torch.vstack([affine_matrix, torch.tensor([0.0, 0.0, 1.0])])

    # Input image: ensure float for resampling, remember original dtype
    orig_dtype = image.dtype
    img = image.to(dtype=torch.float32, device='cpu')  # (1, H, W)
    H, W = output_size
    inH, inW = img.shape[-2], img.shape[-1]

    # Build K matrices for normalized <-> pixel coords with align_corners=True
    def build_K(h, w):
        sx = (w - 1) / 2.0
        sy = (h - 1) / 2.0
        K = torch.tensor([[sx, 0.0, sx],
                          [0.0, sy, sy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        Kinv = torch.tensor([[2.0 / (w - 1), 0.0, -1.0],
                             [0.0, 2.0 / (h - 1), -1.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
        return K, Kinv

    K_out, _ = build_K(H, W)
    _, Kinv_in = build_K(inH, inW)

    # theta maps OUTPUT normalized coords -> INPUT normalized coords
    theta_3x3 = Kinv_in @ affine_matrix @ K_out
    theta = theta_3x3[:2, :].unsqueeze(0)  # (1, 2, 3)

    # Prepare tensors for grid_sample
    img_bchw = img.unsqueeze(0)  # (1, 1, inH, inW)
    grid = torch.nn.functional.affine_grid(theta, size=(1, 1, H, W), align_corners=True)
    out = torch.nn.functional.grid_sample(img_bchw, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    out = out.squeeze(0)  # (1, H, W)

    # Cast back to original dtype if needed
    if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        out = out.clamp(0.0, 255.0).round().to(dtype=torch.uint8)
    return out
from utils import affine_3x3_to_2x3, affine_2x3_to_3x3, get_sift_features, tensor2numpy, numpy2tensor


def preprocess_image(img_path, do_sift=False):
    """
    读取图像路径，进行一系列变换之后返回图像的numpy数组
    :param do_sift: 是否进行特征点检测
    :param img_path: 图像的路径
    :return: 变换后的图像numpy数组，和kps（如果do_sift为True）
    """
    # 以cv2.IMREAD_UNCHANGED标志读取图像，以保留alpha通道
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 4:
        # 直接将 BGRA 转为 BGR，速度更快
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # 如果不包含alpha通道，直接使用读取的图像

    # 将图像转换为3通道，去除alpha通道
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    # img = cv2.blur(img, (4, 4))

    # 自适应阈值
    # _, img_thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 双边滤波
    # img_trans = cv2.bilateralFilter(img_thresh, 15, 0, 255)

    # 归一化到[0, 1]
    # img_trans = img_trans / 255.0

    if do_sift:
        kps = get_sift_features(img)
        return img, kps
    else:
        return img


def resize_image(img, target_size=(256, 256)):
    """
    :param img: 输入的图像numpy数组
    :param target_size:目标的变换大小
    :return:变换之后的图像numpy数组
    """
    # 将图像调整到目标尺寸
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def apply_affine_transform(img, matrix):
    """
    先应用变换，再应用resize
    :param img: 需要进行变化的图像numpy数组
    :param matrix: 进行变换的变换矩阵
    :return:经过变换之后的图片numpy数组
    """
    # 应用仿射变换
    transformed_img = cv2.warpAffine(img,
                                     matrix,
                                     (1024, 1024),  # 原始大小，为了配合gt变换矩阵
                                     # flags=cv2.INTER_LINEAR,
                                     borderValue=(0, 0, 0))
    return transformed_img


def get_file_dictionary(folder_path):
    """返回一个字典，其中包括所有图像文件的文件名和它们的完整路径"""
    file_dict = {}
    # 定义支持的图像文件扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.tif')
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的扩展名，并检查是否是有效的图像文件
            _, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions:
                file_path = os.path.join(root, file)
                file_dict[file] = file_path
    
    return file_dict

def generate_random_affine_matrix(scale_variation=0.2, rotation_variation=30, translation_variation=60):
    """
    Generate a random affine transformation matrix that includes scaling, rotation,
    and translation only.

    :param scale_variation: Maximum percentage variation of scale (0.1 for 10%)
    :param rotation_variation: Maximum rotation in degrees
    :param translation_variation: Max translation in pixels
    :return: A 3x3 affine transformation matrix
    """
    # 随机生成缩放因子，1 ± scale_variation
    sx = 1 + np.random.uniform(-scale_variation, scale_variation)

    # 随机生成旋转角度
    rotation = np.random.uniform(-rotation_variation, rotation_variation)
    theta = np.radians(rotation)

    # 随机生成平移参数
    tx = np.random.uniform(-translation_variation, translation_variation)
    ty = np.random.uniform(-translation_variation, translation_variation)

    # 构建旋转矩阵
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # 构建缩放矩阵
    scale_matrix = np.array([
        [sx, 0, 0],
        [0, sx, 0],
        [0, 0, 1]
    ])

    # 构建平移矩阵
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    # 组合矩阵：先缩放，再旋转，最后平移
    affine_matrix = translation_matrix @ rotation_matrix @ scale_matrix
    affine_matrix = affine_3x3_to_2x3(affine_matrix)

    return affine_matrix


def preprocess_all_images(he_folder_path, cdx_folder_path, target_size=(256, 256), rand_cfg=None):
    """
    :param cdx_folder_path: cdx图像的路径
    :param he_folder_path: he图像的路径
    :param target_size:
    :return:
    """
    all_img_data = []
    he_files = get_file_dictionary(he_folder_path)
    # for key, value in he_files.items():
        # print(key, ":", value)
    cdx_files = get_file_dictionary(cdx_folder_path)
    # for key, value in cdx_files.items():
    #     print(key, ":", value)

    for file_name in he_files:
        if file_name in cdx_files:
            # print("test")
            # 得到图像路径s
            img1_path = file_name
            img2_path = file_name
            # 加载1024原图
            img1 = preprocess_image(os.path.join(he_folder_path, img1_path))
            img2, kps = preprocess_image(os.path.join(cdx_folder_path, img2_path), do_sift=True)
            # 变换
            matrix = generate_random_affine_matrix(**(rand_cfg or {}))
            img2_transformed = apply_affine_transform(img2, matrix)

            # resize到同一个大小
            img1_resized = resize_image(img1, target_size)
            img2_resized = resize_image(img2, target_size)
            img2_transformed_resized = resize_image(img2_transformed, target_size)
            matrix = affine_2x3_to_3x3(matrix)
            matrix_inv = np.linalg.inv(matrix)
            # matrix_inv = matrix

            scale_m = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]])
            matrix = np.linalg.inv(scale_m @ np.linalg.inv(scale_m @ matrix_inv))
            matrix_tenser = torch.from_numpy(matrix).float()

            # 在最后将numpy数组转换为张量
            img1_resized_tensor = torch.from_numpy(img1_resized).unsqueeze(0).float()
            img2_resized_tensor = torch.from_numpy(img2_resized).unsqueeze(0).float()
            img2_transformed_resized_tensor = torch.from_numpy(img2_transformed_resized).unsqueeze(0).float()
            # 将数据组织成所需的格式
            all_img_data.append((
                img1_resized_tensor,  # reference_img
                img2_transformed_resized_tensor,  # floating_img
                img2_resized_tensor,  # ground_truth_img
                matrix_tenser,  # ground_truth_matrix
                kps  # floating_img_kps_np
            ))

    return all_img_data


def generate_affine_matrix_fixed(one_hot_action, translation=1, rotation=1, scale=0.01, device='cpu'):
    """
    Generate an affine matrix corresponding to a specified action with fixed values.

    Args:
        one_hot_action (torch.Tensor): One-hot encoded action (num_actions).
        translation : Fixed translation values (x, y).
        rotation (float): Fixed rotation angle in degrees.
        scale (float): Fixed scale factor.
        device (str): Device to perform computation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Affine transformation matrix (3 x 3) on the specified device.
    """
    width, height = [256, 256]  # 你需要提供图像的宽度和高度

    # 计算图像中心的坐标
    center_x = width / 2
    center_y = height / 2

    matrix = torch.eye(3, device=device)  # Identity matrix on the correct device

    # Apply transformations based on one-hot encoded action
    if one_hot_action == 0:  # Downward translation
        matrix[1, 2] = -translation
    elif one_hot_action == 1:  # Upward translation
        matrix[1, 2] = translation
    elif one_hot_action == 2:  # Rightward translation
        matrix[0, 2] = -translation
    elif one_hot_action == 3:  # Leftward translation
        matrix[0, 2] = translation
    elif one_hot_action == 4:  # Clockwise rotation
        angle_rad = -math.pi * rotation / 180
        # 创建一个平移矩阵，先平移到图像中心，然后旋转，最后平移回原位置
        matrix = torch.tensor([[1, 0, center_x],
                               [0, 1, center_y],
                               [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                 [math.sin(angle_rad), math.cos(angle_rad), 0],
                                 [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1, 0, -center_x],
                                 [0, 1, -center_y],
                                 [0, 0, 1]], dtype=torch.float32, device=device)
    elif one_hot_action == 5:  # Counterclockwise rotation
        angle_rad = math.pi * rotation / 180
        # 与上面相同的操作，只是旋转角度不同
        matrix = torch.tensor([[1, 0, center_x],
                               [0, 1, center_y],
                               [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                 [math.sin(angle_rad), math.cos(angle_rad), 0],
                                 [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1, 0, -center_x],
                                 [0, 1, -center_y],
                                 [0, 0, 1]], dtype=torch.float32, device=device)
    elif one_hot_action == 6:  # 缩小
        matrix = torch.tensor([[1, 0, center_x],
                               [0, 1, center_y],
                               [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1 - scale, 0, 0],
                                 [0, 1 - scale, 0],
                                 [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1, 0, -center_x],
                                 [0, 1, -center_y],
                                 [0, 0, 1]], dtype=torch.float32, device=device)

    elif one_hot_action == 7:  # 放大
        matrix = torch.tensor([[1, 0, center_x],
                               [0, 1, center_y],
                               [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1 + scale, 0, 0],
                                 [0, 1 + scale, 0],
                                 [0, 0, 1]], dtype=torch.float32, device=device) \
                 @ torch.tensor([[1, 0, -center_x],
                                 [0, 1, -center_y],
                                 [0, 0, 1]], dtype=torch.float32, device=device)

    return matrix


def apply_affine_transform_cv2(image, affine_matrix):
    """
    Apply an affine transformation to an image using OpenCV.

    Args:
        image (np.array): The image to transform (height x width x channels).
        affine_matrix (np.array): The affine transformation matrix (3x3).

    Returns:
        np.array: The transformed image.
    """

    # Convert the image to a numpy array if it isn't one already

    image = tensor2numpy(image)[0]
    affine_matrix = tensor2numpy(affine_matrix)

    # Ensure the affine matrix is a 2x3 matrix
    if affine_matrix.shape[0] == 3:
        affine_matrix = affine_3x3_to_2x3(affine_matrix)

    # print(affine_matrix)
    # print(image.shape)
    # Get the image dimensions
    height, width = image.shape[:2]

    # Apply the affine transformation
    transformed_image = cv2.warpAffine(
        image,
        affine_matrix,  # Use only the 2x3 part of the matrix
        (width, height),  # Output image size
        # flags=cv2.INTER_LINEAR
        borderValue=(0, 0, 0)
    )

    transformed_image = numpy2tensor(transformed_image).unsqueeze(0)

    return transformed_image


def move(action_index, image, matrix, action_mats=None):
    """
    该函数接受变换对应的编码和图像，返回变换后的图像
    :param action_index: 代表动作的编码
    :param image: 需要进行变换的图像
    :param matrix: 需要进行累乘的矩阵,该矩阵累乘之后才是我们需要用的矩阵，因为输入图像是没有经过任何变换的初试图像
    :return move_image: 经过变换的图像
    :return move_matrix: 经过累乘的矩阵
    """
    # 确保图像和矩阵都在正确的设备上
    image = image
    initial_matrix = matrix
    if action_mats is None:
        affine_matrix = generate_affine_matrix_fixed(action_index)
    else:
        affine_matrix = action_mats[action_index]
    move_matrix = torch.matmul(affine_matrix, initial_matrix)

    # 应用仿射变换到图像上
    move_image = apply_affine_transform_cv2(image, move_matrix)

    return move_image, move_matrix
