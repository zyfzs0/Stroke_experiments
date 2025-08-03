import os
import numpy as np
import cv2

def process_strokes():
    # 设置文件夹路径
    base_dir = "./"
    folders = ["sdnet_blank", "sdnet_kaiti", "strokediff", "mask_cnn"]
    output_names = ["result_blank.png", "result_kaiti.png", "result_strokediff.png", "result_mask_cnn.png"]
    
    # 20种颜色设置 (BGR格式)
    colors = [
        (0, 0, 255),       # 红色
        (0, 255, 0),       # 绿色
        (255, 0, 0),       # 蓝色
        (0, 255, 255),     # 黄色
        (255, 0, 255),     # 紫色
        (255, 255, 0),     # 青色
        (0, 128, 255),     # 橙色
        (128, 0, 255),     # 粉紫色
        (0, 255, 128),     # 春绿色
        (128, 255, 0),     # 酸橙色
        (255, 0, 128),     # 深粉色
        (255, 128, 0),     # 深橙色
        (0, 128, 128),     # 蓝绿色
        (128, 0, 128),     # 靛蓝色
        (128, 128, 0),     # 橄榄色
        (192, 192, 192),   # 银色
        (128, 128, 128),   # 灰色
        (64, 0, 0),        # 深红色
        (0, 64, 0),        # 深绿色
        (0, 0, 64)         # 深蓝色
    ]
    
    # 检查文件夹是否存在
    for folder in folders:
        if not os.path.exists(os.path.join(base_dir, folder)):
            print(f"错误: 文件夹 {folder} 不存在")
            return
    
    # 获取所有笔画文件并按笔画编号排序
    stroke_files = []
    for folder in folders:
        files = os.listdir(os.path.join(base_dir, folder))
        files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
        # 按文件名中的数字排序（假设文件名格式为"stroke_1.png"）
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        stroke_files.append(files)
    
    # 检查文件数量是否一致
    if len(set(len(files) for files in stroke_files)) != 1:
        print("错误: 文件夹中的笔画数量不一致")
        return
    
    # 初始化四个空白画布 (黑色背景)，尺寸为256x256
    canvas_size = (256, 256)
    canvases = [np.zeros((*canvas_size, 3), dtype=np.uint8) for _ in range(4)]
    
    # 按顺序处理每个笔画
    for i in range(len(stroke_files[0])):
        color = colors[i % len(colors)]  # 循环使用颜色
        
        for canvas_idx in range(4):
            folder = folders[canvas_idx]
            file = stroke_files[canvas_idx][i]
            file_path = os.path.join(base_dir, folder, file)
            
            # 读取图像
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"警告: 无法读取图像 {file_path}, 将使用空白图像代替")
                img = np.ones(canvas_size, dtype=np.uint8) * 255
            else:
                img = cv2.resize(img, canvas_size)
            
            # 对mask_cnn文件夹特殊处理
            if folder == "mask_cnn":
                # 反转颜色 (白色变黑色，黑色变白色)
                img = 255 - img
                _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
                # 使用相同的颜色
                colored_stroke = np.zeros((*img.shape, 3), dtype=np.uint8)
                colored_stroke[binary == 0] = color
            else:
                # 常规处理
                _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
                colored_stroke = np.zeros((*img.shape, 3), dtype=np.uint8)
                colored_stroke[binary == 0] = color
            
            # 直接覆盖像素值
            canvases[canvas_idx][colored_stroke.any(axis=-1)] = colored_stroke[colored_stroke.any(axis=-1)]
    
    # 保存结果并反转颜色
    for canvas_idx in range(4):
        output_name = output_names[canvas_idx]
        # 对所有结果进行颜色反转
        canvas = 255 -canvases[canvas_idx]
        # 保存图像
        cv2.imwrite(os.path.join(base_dir, output_name), canvas)
        print(f"已保存: {output_name}")

if __name__ == "__main__":
    process_strokes()