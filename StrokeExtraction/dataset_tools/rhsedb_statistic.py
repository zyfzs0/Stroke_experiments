

import os
import numpy as np
import pandas as pd
from pathlib import Path

def process_npz_files():
    # 定义目录路径
    test_dir = "/home/tongji209/majiawei/Stroke_experiments/RHSEDB/test"
    train_dir = "/home/tongji209/majiawei/Stroke_experiments/RHSEDB/train"
    
    # 准备收集数据的列表
    data_list = []
    
    # 遍历test目录
    for npz_file in Path(test_dir).rglob('*.npz'):
        process_single_file(npz_file, 'test', data_list)
    
    # 遍历train目录
    for npz_file in Path(train_dir).rglob('*.npz'):
        process_single_file(npz_file, 'train', data_list)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data_list, columns=['unicode', 'stroke_count', 'dataset', 'npz_name'])
    df.to_csv('/home/tongji209/majiawei/Stroke_experiments/RHSEDB/character_info.csv', index=False, encoding='utf-8-sig')
    print("CSV文件已生成: character_info.csv")

def process_single_file(npz_path, dataset, data_list):
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # 获取汉字字符
        char_name = data['name'].item() if isinstance(data['name'], np.ndarray) else data['name']
        
        # 转换为Unicode码点
        unicode_value = ord(char_name)
        
        # 获取笔画数
        stroke_data = data['stroke_label']
        if isinstance(stroke_data, (np.ndarray, list)):
            stroke_count = len(stroke_data)
        else:
            stroke_count = 1 
        
        # 获取npz文件名
        npz_name = os.path.basename(npz_path)
        
        # 添加到数据列表
        data_list.append({
            'unicode': unicode_value,
            'stroke_count': stroke_count,
            'dataset': dataset,
            'npz_name': npz_name
        })
        
    except Exception as e:
        print(f"处理文件 {npz_path} 时出错: {str(e)}")

if __name__ == "__main__":
    process_npz_files()


