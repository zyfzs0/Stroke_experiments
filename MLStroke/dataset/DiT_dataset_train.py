import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CharacterStrokePairDatasetTrain(Dataset):
    def __init__(self,split,cfg,
                 csv_path: str,
                 character_dir: str,
                 character_suffix: str = ".jpg",
                 stroke_suffix: str = ".jpg",
                 character_transform=None,
                 stroke_transform=None,
                 ):
        """
        参数说明：
        - csv_path: CSV文件路径
        - character_dir: 完整汉字图片目录（存放如11904.png）
        - character_suffix: 汉字图片后缀
        - stroke_suffix: 笔画图片后缀
        - transform: 汉字图片的预处理
        - stroke_transform: 笔画图片的预处理
        """
        self.split = split
        self.cfg = cfg

        self.df = pd.read_csv(csv_path)
        self.character_dir = character_dir
        self.character_suffix = character_suffix
        self.stroke_suffix = stroke_suffix

        # 设置默认转换
        self.character_transform = character_transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.stroke_transform = stroke_transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        # 预验证数据有效性
        self.valid_indices = []
        # print(len(self.df))
        for idx in range(len(self.df)):
            char_path = self._get_character_path(idx)
            stroke_path = self._get_stroke_path(idx)
            # print(char_path,stroke_path)
            if os.path.exists(char_path) and os.path.exists(stroke_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Missing files at index {idx}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        # 获取路径
        char_path = self._get_character_path(real_idx)
        stroke_path = self._get_stroke_path(real_idx)

        # 加载图像
        char_img = Image.open(char_path).convert('RGB')
        stroke_img = Image.open(stroke_path).convert('RGB')  # 笔画通常用灰度

        if self._strokes_pre_valid(real_idx):
            # 如果有笔画预处理图片，加载它
            strokes_pre_path = self._get_strokes_pre_path(real_idx)
            strokes_pre_img = Image.open(strokes_pre_path).convert('RGB')
        else:
            width, height = char_img.size
            strokes_pre_img = Image.new('RGB', (width, height), (255, 255, 255))

        # 应用转换
        if self.character_transform:
            char_img = self.character_transform(char_img)
        if self.stroke_transform:
            stroke_img = self.stroke_transform(stroke_img)
            strokes_pre_img = self.stroke_transform(strokes_pre_img)

        return {
            'character': char_img,
            'unicode': torch.tensor(self.df.iloc[real_idx]['unicode']),
            'stroke': stroke_img,
            # 'character_id': self.df.iloc[real_idx]['character'],
            'stroke_order': torch.tensor(self.df.iloc[real_idx]['stroke'] - 1),
            'stroke_num': torch.tensor(self.df.iloc[real_idx]['stroke_nums'] - 1),
            'strokes_pre': strokes_pre_img,

            # 'target': self.df.iloc[real_idx]['target']
        }