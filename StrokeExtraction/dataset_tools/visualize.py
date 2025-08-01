import numpy as np
from PIL import Image
import os


def save_npz_images(npz_path, output_dir):
    """
    将 NPZ 文件中的图像数据保存为图片文件

    参数:
        npz_path: NPZ 文件路径
        output_dir: 输出图片的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载 NPZ 文件
    data = np.load(npz_path)
    print(data['name'])
    print(data['stroke_name'])
    print(data['stroke_label'])

    # 保存参考彩色图像 (reference_color_image)
    if 'reference_color_image' in data:
        ref_color = data['reference_color_image']
        if ref_color.dtype == np.float32 or ref_color.dtype == np.float64:
            ref_color = (ref_color * 255).astype(np.uint8)
        # 转换形状: (3, 256, 256) -> (256, 256, 3)
        ref_color_img = Image.fromarray(ref_color.transpose(1, 2, 0))
        ref_color_img.save(os.path.join(output_dir, "ref_color.png"))

    # 保存参考单笔画图像 (reference_single_image)
    if 'reference_single_image' in data:
        ref_singles = data['reference_single_image']
        for i, stroke in enumerate(ref_singles):
            if stroke.dtype == np.float32 or stroke.dtype == np.float64:
                stroke = (stroke * 255).astype(np.uint8)
            stroke_img = Image.fromarray(stroke).convert('L')  # 灰度图
            stroke_img.save(os.path.join(output_dir, f"ref_stroke_{i}.png"))

    # 保存目标整体图像 (target_image)
    if 'target_image' in data:
        target_img_data = data['target_image'][0]  # (1,256,256) -> (256,256)
        if target_img_data.dtype == np.float32 or target_img_data.dtype == np.float64:
            target_img_data = (target_img_data * 255).astype(np.uint8)
        target_img = Image.fromarray(target_img_data).convert('L')
        target_img.save(os.path.join(output_dir, "target.png"))

    # 保存目标单笔画图像 (target_single_image)
    if 'target_single_image' in data:
        target_singles = data['target_single_image']
        for i, stroke in enumerate(target_singles):
            if stroke.dtype == np.float32 or stroke.dtype == np.float64:
                stroke = (stroke * 255).astype(np.uint8)
            stroke_img = Image.fromarray(stroke).convert('L')
            stroke_img.save(os.path.join(output_dir, f"target_stroke_{i}.png"))


# 使用示例
if __name__ == "__main__":
    save_npz_images(r"/home/tongji209/majiawei/Stroke_experiments/RHSEDB/test/857.npz", r"/home/tongji209/majiawei/Stroke_experiments/RHSEDB/visual")