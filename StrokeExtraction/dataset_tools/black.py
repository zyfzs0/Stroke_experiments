import numpy as np
import cv2

# 创建一个256x256的白色图像（三通道）
white_image = np.zeros((256, 256, 3), dtype=np.uint8) * 255

# 保存为PNG文件
cv2.imwrite('white_image.png', white_image)

print("已生成全白图像 white_image.png")