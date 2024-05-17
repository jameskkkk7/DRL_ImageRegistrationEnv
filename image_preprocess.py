import math
import os
import re
import cv2
import numpy as np
import torch
from utils import affine_3x3_to_2x3, affine_2x3_to_3x3, get_sift_features, tensor2numpy, numpy2tensor


def load_path_dic(file_path):
    data_dict = {}
    with open(file_path, "r", encoding="GBK") as file:
        for line in file:
            match_img1 = re.search(r"图片1名称: ([^,]+)", line)
            match_img2 = re.search(r"图片2名称: ([^,]+)", line)
            match_matrix = re.search(r"Affine matrix: (\[\[.*?\]\])", line)

            if match_img1 and match_img2 and match_matrix:
                folder_name = match_img1.group(1).split("_")[0]
                img_name1 = os.path.join(
                    "HE_image", folder_name, match_img1.group(1).strip() + ".tif"
                )
                img_name2 = os.path.join(
                    "CDX_image", folder_name, match_img2.group(1).strip() + ".tif"
                )

                matrix_str = match_matrix.group(1)
                matrix_list = eval(matrix_str)
                matrix_array = np.array(matrix_list)

                if matrix_array.shape == (3, 3):
                    matrix_array = matrix_array[:2, :3]

                data_dict[img_name1] = {
                    "img_name2": img_name2,
                    "matrix": matrix_array,
                }
    return data_dict


def preprocess_image(img_path, do_sift=False):
    """
    读取图像路径，进行一系列变换之后返回图像的numpy数组
    :param do_sift: 是否进行特征点检测
    :param img_path: 图像的路径
    :return: 变换后的图像numpy数组，和kps
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_trans = cv2.blur(img, (4, 4))  # 高斯模糊
    img_trans = cv2.adaptiveThreshold(img_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img_trans = cv2.bilateralFilter(img_trans, 15, 0, 255)
    img_trans = img_trans / 255.0
    if do_sift:
        kps = get_sift_features(img)
        return img_trans, kps
    else:
        return img_trans


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
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    return transformed_img


def preprocess_all_images(root_folder, txt_file_path, target_size=(256, 256)):
    """
    :param root_folder:
    :param txt_file_path:
    :param target_size:
    :return:
    """
    data_dict = load_path_dic(txt_file_path)
    img_name1_list = list(data_dict.keys())
    all_img_data = []

    for img_name1 in img_name1_list:
        img_name2 = data_dict[img_name1]['img_name2']
        matrix = data_dict[img_name1]['matrix']
        # 加载1024原图
        img1 = preprocess_image(os.path.join(root_folder, img_name1))
        img2, kps = preprocess_image(os.path.join(root_folder, img_name2), do_sift=True)
        # 变换
        img2_transformed = apply_affine_transform(img2, matrix)

        # resize到同一个大小
        img1_resized = resize_image(img1, target_size)
        img2_resized = resize_image(img2, target_size)
        img2_transformed_resized = resize_image(img2_transformed, target_size)

        # 在最后将numpy数组转换为张量
        img1_resized_tensor = torch.from_numpy(img1_resized).unsqueeze(0).float()
        img2_resized_tensor = torch.from_numpy(img2_resized).unsqueeze(0).float()
        img2_transformed_resized_tensor = torch.from_numpy(img2_transformed_resized).unsqueeze(0).float()
        matrix = affine_2x3_to_3x3(matrix)
        scale_m = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]])
        matrix = np.linalg.inv(scale_m @ np.linalg.inv(scale_m @ matrix))
        matrix_tenser = torch.from_numpy(matrix).float()
        # 将数据组织成所需的格式
        all_img_data.append((
            img1_resized_tensor,  # reference_img
            img2_resized_tensor,  # floating_img
            img2_transformed_resized_tensor,  # ground_truth_img
            matrix_tenser,  # ground_truth_matrix
            kps  # floating_img_kps_np
            # TODO:这里的matrix需要做变换，才能符合被resize之后的图片,依旧有问题
        ))

    return all_img_data


# ----------------------------------------------------------------------------------------------------------------------
# 以上部分运算由CPU执行，以下的运算必须由GPU执行


def generate_affine_matrix_fixed(one_hot_action, translation=2, rotation=2, scale=0.1, device='cpu'):
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
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR  # Use inverse mapping and bilinear interpolation
    )

    transformed_image = numpy2tensor(transformed_image).unsqueeze(0)

    return transformed_image


def move(action_index, image, matrix):
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
    affine_matrix = generate_affine_matrix_fixed(action_index)
    move_matrix = torch.matmul(affine_matrix, initial_matrix)

    # 应用仿射变换到图像上
    move_image = apply_affine_transform_cv2(image, move_matrix)

    return move_image, move_matrix
