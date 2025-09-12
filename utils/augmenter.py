
import os
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class ReproducibleAugmenter:
    """
    一个可复现的数据增强器。

    通过设置固定的随机数种子，确保每次运行时对同一数据集的增强结果
    (应用的变换、参数和最终顺序) 完全一致。

    Attributes:
        seed (int): 用于所有随机操作的种子。
        num_augmentations (int): 每张原始图片要生成的增强版本的数量。
        use_rotation (bool): 是否启用随机旋转。
        use_cropping (bool): 是否启用随机裁剪和缩放。
        use_brightness (bool): 是否启用随机亮度调整。
    """

    def __init__(self, seed, num_augmentations_per_image, use_rotation=True, use_cropping=True, use_brightness=True):
        """
        初始化增强器。

        Args:
            seed (int): 随机数种子。
            num_augmentations_per_image (int): 为每张原始图片生成的增强图片的数量。
            use_rotation (bool, optional): 是否应用旋转。默认为 True。
            use_cropping (bool, optional): 是否应用裁剪。默认为 True。
            use_brightness (bool, optional): 是否应用亮度调整。默认为 True。
        """
        self.seed = seed
        self.num_augmentations = num_augmentations_per_image
        self.use_rotation = use_rotation
        self.use_cropping = use_cropping
        self.use_brightness = use_brightness
        print(f"增强器已初始化。种子: {self.seed}, 每张图片生成 {self.num_augmentations} 个增强版本。")

    def _rotate_image(self, image, angle):
        """旋转图像，保持尺寸不变"""
        height, width = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated

    def _adjust_brightness(self, img):
        """使用gamma变换调整亮度"""
        mu = 0
        sigma = 0.3
        gamma = np.random.lognormal(mu, sigma)
        img_normalized = img / 255.0
        img_gamma = np.power(img_normalized, gamma)
        img = (img_gamma * 255).astype(np.uint8)
        return img

    def _resize_img(self, img, target_size=(1024, 1024)):
        """调整图像大小到目标尺寸"""
        return cv2.resize(img, target_size)

    def _apply_single_augmentation(self, image):
        """对单张图像应用已启用的随机变换"""
        aug_image = image.copy()

        # 1. 随机旋转
        if self.use_rotation:
            angle = random.randint(-30, 30)
            aug_image = self._rotate_image(aug_image, angle)
        
        # 2. 随机裁剪后resize
        if self.use_cropping:
            crop_size = random.randint(800, 1000)
            start_x = random.randint(0, 1024 - crop_size)
            start_y = random.randint(0, 1024 - crop_size)
            aug_image = aug_image[start_y:start_y+crop_size, start_x:start_x+crop_size]
            aug_image = self._resize_img(aug_image)
        
        # 3. gamma变换调整亮度
        if self.use_brightness:
            aug_image = self._adjust_brightness(aug_image)

        return aug_image

    def augment(self, data_list, is_positive_only=False, debug_save_path=None):
        """
        对整个数据列表进行增强。

        Args:
            data_list (list): 原始数据列表，每个元素是一个字典。
            is_positive_only (bool, optional): 是否只对正样本 (positive==1) 进行增强。默认为 False。
            debug_save_path (str, optional): 如果提供，将保存一些增强图像用于调试。默认为 None。

        Returns:
            list: 包含原始数据和增强后数据的最终列表，并已进行确定性随机打乱。
        """
        # !!! 关键步骤：在函数开始时设置种子，确保所有后续操作可复现 !!!
        random.seed(self.seed)
        np.random.seed(self.seed)

        if debug_save_path:
            os.makedirs(debug_save_path, exist_ok=True)

        augmented_data = []
        print(f"开始对 {len(data_list)} 条数据进行增强...")

        for data in tqdm(data_list, desc="Augmenting Data"):
            # 首先，无条件添加原始数据
            augmented_data.append(data)

            # 检查是否需要对当前数据进行增强
            if self.num_augmentations > 0:
                if is_positive_only and data.get("positive", 0) != 1:
                    continue  # 如果只增强正样本，而当前是负样本，则跳过

                for i in range(self.num_augmentations):
                    # 应用增强
                    img_aug = self._apply_single_augmentation(data["img"])
                    
                    # 复制元数据并创建新的数据条目
                    aug_data_item = data.copy() # 使用copy()来避免修改原始字典
                    aug_data_item["img"] = img_aug
                    aug_data_item["file_name"] = data["file_name"] + f"_aug_{i}"
                    
                    augmented_data.append(aug_data_item)

                    # 保存部分增强图像用于调试
                    if debug_save_path and random.random() < 0.05: # 只保存约5%
                        Image.fromarray(img_aug).save(
                            os.path.join(debug_save_path, f"{os.path.splitext(data['file_name'])[0]}_aug_{i}.png")
                        )
        
        # !!! 关键步骤：使用已设置种子的random库进行打乱，保证顺序固定 !!!
        random.shuffle(augmented_data)
        
        print(f"增强完成。原始数据量: {len(data_list)}, 增强后总数据量: {len(augmented_data)}")
        return augmented_data