import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EvaluateDataset(Dataset):
    def __init__(self, 
                 csv_path: str,
                 ):

        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            
        return {
            'unicode':torch.tensor(self.df.iloc[idx]['unicode']),
            'stroke_num': torch.tensor(self.df.iloc[idx]['stroke_nums']-1),
        }



class CharacterStrokePairDataset(Dataset):
    def __init__(self, 
                 csv_path: str,
                 character_dir: str,
                 stroke_dir: str,
                 character_suffix: str = ".jpg",
                 stroke_suffix: str = ".jpg",
                 character_transform=None,
                 stroke_transform=None):
        """
        参数说明：
        - csv_path: CSV文件路径
        - character_dir: 完整汉字图片目录（存放如11904.png）
        - stroke_dir: 笔画图片目录（存放如11904_1.png）
        - character_suffix: 汉字图片后缀
        - stroke_suffix: 笔画图片后缀
        - transform: 汉字图片的预处理
        - stroke_transform: 笔画图片的预处理
        """
        self.df = pd.read_csv(csv_path)
        self.character_dir = character_dir
        self.stroke_dir = stroke_dir
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

    def _get_character_path(self, idx: int) -> str:
        """构建汉字图片路径"""
        char_id = str(self.df.iloc[idx]['character'])
        return os.path.join(
            self.character_dir, 
            f"{char_id}{self.character_suffix}"
        )

    def _get_stroke_path(self, idx: int) -> str:
        """构建笔画图片路径"""
        target = str(self.df.iloc[idx]['target'])
        return os.path.join(
            self.stroke_dir,
            f"{target}{self.stroke_suffix}"
        )
        
    def _get_strokes_pre_path(self, idx: int) -> str:
        strokes_pre = str(self.df.iloc[idx]['strokes_pre'])
        return os.path.join(
            self.stroke_dir,
            f"{strokes_pre}{self.stroke_suffix}"
        )
        
    def _strokes_pre_valid(self, idx: int) -> bool:
        """检查笔画预处理图片是否存在"""
        strokes_pre_path = self._get_strokes_pre_path(idx)
        return os.path.exists(strokes_pre_path)

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
            'unicode':torch.tensor(self.df.iloc[real_idx]['unicode']),
            'stroke': stroke_img,
            # 'character_id': self.df.iloc[real_idx]['character'],
            'stroke_order': torch.tensor(self.df.iloc[real_idx]['stroke']-1),
            'stroke_num': torch.tensor(self.df.iloc[real_idx]['stroke_nums']-1),
            'strokes_pre': strokes_pre_img,

            # 'target': self.df.iloc[real_idx]['target']
        }

# 使用示例
if __name__ == "__main__":
    # 示例转换配置
    char_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    stroke_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = CharacterStrokePairDataset(
        csv_path="/home/tongji209/majiawei/stroke_segmentation/stroke_data.csv",
        character_dir="/home/tongji209/majiawei/stroke_segmentation/pixel_all_characters",
        stroke_dir="/home/tongji209/majiawei/stroke_segmentation/pixel_all_strokes",
        character_transform=char_transform,
        stroke_transform=stroke_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: {
            'character': torch.stack([x['character'] for x in batch]),
            'stroke': torch.stack([x['stroke'] for x in batch]),
            # 'character_id': [x['character_id'] for x in batch],
            'stroke_order': torch.tensor([x['stroke_order'] for x in batch]),
            'stroke_nums': torch.tensor([x['stroke_nums'] for x in batch]),
            # 'target': [x['target'] for x in batch]
        }
    )

    for batch in dataloader:
        print(batch['character'].size())
        print(batch['stroke'].size())
        print(batch['stroke_order'].size())
        print(batch['stroke_nums'].size())
        input()

    # 测试数据加载
    sample = next(iter(dataloader))
    print(f"Character tensor shape: {sample['character'].shape}")  # [32, 3, 128, 128]
    print(f"Stroke tensor shape: {sample['stroke'].shape}")        # [32, 1, 64, 64]
    print(f"Stroke orders: {sample['stroke_order'][:5]}")          # 展示前5个笔画序号




