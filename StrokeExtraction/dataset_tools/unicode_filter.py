
import pandas as pd

# 读取character_info.csv文件获取有效的Unicode列表
character_info = pd.read_csv('character_info.csv')
valid_unicodes = set(character_info['unicode'].unique())

def filter_file(input_filename, output_filename):
    # 读取输入文件
    df = pd.read_csv(input_filename)
    
    # 过滤数据，只保留unicode在valid_unicodes中的行
    filtered_df = df[df['unicode'].isin(valid_unicodes)]
    
    # 保存到新文件
    filtered_df.to_csv(output_filename, index=False)
    print(f"已处理文件 {input_filename}，保留 {len(filtered_df)} 条记录，保存为 {output_filename}")

# 处理test文件
filter_file('test_metadata.csv', 'test_metadata_filter.csv')

# 处理test文件
filter_file('test_metadata_with_pre.csv', 'test_metadata_with_pre_filter.csv')

# 处理train文件
filter_file('train_metadata.csv', 'train_metadata_filter.csv')
# 处理train文件
filter_file('train_metadata_with_pre.csv', 'train_metadata_with_pre_filter.csv')