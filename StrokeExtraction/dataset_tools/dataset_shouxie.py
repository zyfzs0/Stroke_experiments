import os
import numpy as np
from PIL import Image
import pandas as pd

# 配置路径
input_csv = 'character_info.csv'  # 替换为实际CSV路径
output_char_dir = 'output/characters'
output_stroke_dir = 'output/strokes'
os.makedirs(output_char_dir, exist_ok=True)
os.makedirs(output_stroke_dir, exist_ok=True)

# 读取CSV并处理
df = pd.read_csv(input_csv)
records = []

# 取每个unicode的第一个记录
unique_unicodes = df['unicode'].unique()

for unicode in unique_unicodes:
    # 获取该unicode的第一个记录
    row = df[df['unicode'] == unicode].iloc[0]
    npz_path = os.path.join(row['dataset'], row['npz_name'])
    
    try:
        # 加载NPZ文件
        data = np.load(npz_path)
        
        # 处理目标图像
        target_img_data = data['target_image'][0]  # (1,256,256) -> (256,256)
        if target_img_data.dtype in (np.float32, np.float64):
            target_img_data = (target_img_data * 255).astype(np.uint8)
        target_img = Image.fromarray(target_img_data).convert('L')
        target_img.save(os.path.join(output_char_dir, f'{unicode}.png'))
        
        # 处理笔画图像
        target_singles = data['target_single_image']
        stroke_count = target_singles.shape[0]  # 获取实际笔画数
        
        for i in range(stroke_count):
            stroke = target_singles[i]
            if stroke.dtype in (np.float32, np.float64):
                stroke = (stroke * 255).astype(np.uint8)
            stroke_img = Image.fromarray(stroke).convert('L')
            stroke_img.save(os.path.join(output_stroke_dir, f'{unicode}_{i+1}.png'))
            
            # 添加到记录
            records.append({
                'character': unicode,
                'unicode': unicode,
                'stroke': i+1,
                'stroke_nums': stroke_count,
                'target': f'{unicode}_{i+1}'
            })
            
    except Exception as e:
        print(f"处理unicode {unicode}时出错: {str(e)}")
        continue

# 生成DataFrame并保存CSV
result_df = pd.DataFrame(records)
result_df.to_csv('output/dataset_records.csv', index=False)

print(f"处理完成！共处理{len(unique_unicodes)}个唯一字符，生成{len(records)}条笔画记录。")