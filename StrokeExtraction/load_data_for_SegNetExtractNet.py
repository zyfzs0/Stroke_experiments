import os.path
import torch.utils.data as data
import pickle
import time
import torch
import scipy.ndimage as pyimg
import pandas as pd
import os
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import seg_label_to7
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''
load data for SegNet and ExtractNet
'''


class SegNetExtractNetLoader_old(data.Dataset):
    def __init__(self, is_training, dataset_path, is_single=False):
        self.is_training = is_training
        self.is_single = is_single
        if is_training:
            self.path = [[os.path.join(dataset_path, 'train', each),
                          os.path.join(dataset_path, 'train', each[:-16] + '_style.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_single.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_style_single.npy')]
                           for each in os.listdir(os.path.join(dataset_path, 'train')) if 'color' in each]
        else:
            self.path = [[os.path.join(dataset_path, 'test', each),
                          os.path.join(dataset_path, 'test', each[:-16] + '_style.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_single.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_style_single.npy'), int(each[:-16])]
                           for each in os.listdir(os.path.join(dataset_path, 'test')) if
                         'color' in each]

        self.path = sorted(self.path, key=lambda x: x[-1])
        print("number of dataset：%d"%len(self.path))

    def get_seg_image(self, reference_single, seg_label):
        reference_image = np.zeros(shape=(7, 256, 256), dtype=np.float)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            reference_image[id_7] += reference_single[i]
        return np.clip(reference_image, 0, 1)

    def get_data(self, item):
        """
        """
        reference_color = np.load(self.path[item][0])  # (3, 256, 256)
        label_seg = np.load(self.path[item][1])[1:]  # (7, 256, 256)
        target_image = np.load(self.path[item][1])[:1]  # (1, 256, 256)
        seg_id = np.load(self.path[item][2])     # (N)
        reference_transformed_single = np.load(self.path[item][3])  # (N, 256 256)
        target_single_stroke = np.load(self.path[item][4])  # (N, 256 256)
        target_data = np.repeat(target_image, 3, axis=0).astype(np.float)
        reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        label_seg = self.get_seg_image(target_single_stroke, seg_id)


        if self.is_single:  # For ExtractNet
            return {
                'target_data': target_data,
                'reference_color':reference_color,
                'label_seg': label_seg,
                'reference_segment_transformation_data':reference_segment_transformation_data,
                'seg_id': seg_id,
                'reference_transformed_single': reference_transformed_single,
                'target_single_stroke': target_single_stroke

            }
        else:  # For SegNet
            return {

                'target_data': target_data,
                'reference_color': reference_color,
                'label_seg': label_seg
            }

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data


class SegNetExtractNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_single=False,
                 csv_path: str = None,
                 character_dir: str= None,
                 stroke_dir: str= None,
                 character_suffix: str = ".jpg",
                 stroke_suffix: str = ".jpg",
                 character_transform=None,
                 stroke_transform=None):
        self.is_training = is_training
        self.is_single = is_single
        if is_training:
            csv_path = '/remote-home/zhangxinyue/stroke_segmentation/split_data_by_character/train_metadata.csv'

        else:
            csv_path = '/remote-home/zhangxinyue/stroke_segmentation/split_data_by_character/test_metadata.csv'
        
        
        self.df = pd.read_csv(csv_path)
        self.character_dir = '/remote-home/zhangxinyue/stroke_segmentation/pixel_all_characters'
        self.stroke_dir = '/remote-home/zhangxinyue/stroke_segmentation/pixel_all_strokes'
        
        self.character_suffix = character_suffix
        self.stroke_suffix = stroke_suffix
        
         # 设置默认转换
        self.character_transform = character_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.convert('L') if isinstance(x, Image.Image) else x),
            transforms.ToTensor()
        ])
        self.reference_transform = character_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.convert('RGB')),  # 强制转为RGB三通道
            transforms.ToTensor()
        ])
        self.stroke_transform = stroke_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.convert('L') if isinstance(x, Image.Image) else x),
            transforms.ToTensor()
        ])
        
        # 按character分组所有笔画
        self.grouped_df = self.df.groupby('character')
        self.valid_characters = []
        
        # 验证数据有效性
        for char_id, group in self.grouped_df:
            valid = True
            # 检查汉字图片是否存在
            char_path = os.path.join(
                self.character_dir, 
                f"{char_id}{self.character_suffix}"
            )
            if not os.path.exists(char_path):
                print(f"Warning: Missing character image {char_path}")
                valid = False
                continue
                
            # 检查所有笔画图片是否存在
            for _, row in group.iterrows():
                stroke_path = os.path.join(
                    self.stroke_dir,
                    f"{row['target']}{self.stroke_suffix}"
                )
                if not os.path.exists(stroke_path):
                    print(f"Warning: Missing stroke image {stroke_path}")
                    valid = False
                    break
            
            if valid:
                self.valid_characters.append(char_id)
        # if is_training:
        #     self.path = [[os.path.join(dataset_path, 'train', each),
        #                   os.path.join(dataset_path, 'train', each[:-16] + '_style.npy'),
        #                   os.path.join(dataset_path, 'train', each[:-16] + '_seg.npy'),
        #                   os.path.join(dataset_path, 'train', each[:-16] + '_single.npy'),
        #                   os.path.join(dataset_path, 'train', each[:-16] + '_style_single.npy')]
        #                    for each in os.listdir(os.path.join(dataset_path, 'train')) if 'color' in each]
        # else:
        #     self.path = [[os.path.join(dataset_path, 'test', each),
        #                   os.path.join(dataset_path, 'test', each[:-16] + '_style.npy'),
        #                   os.path.join(dataset_path, 'test', each[:-16] + '_seg.npy'),
        #                   os.path.join(dataset_path, 'test', each[:-16] + '_single.npy'),
        #                   os.path.join(dataset_path, 'test', each[:-16] + '_style_single.npy'), int(each[:-16])]
        #                    for each in os.listdir(os.path.join(dataset_path, 'test')) if
        #                  'color' in each]

        # self.path = sorted(self.path, key=lambda x: x[-1])
        # print("number of dataset：%d"%len(self.path))

    def get_seg_image(self, reference_single, seg_label):
        reference_image = np.zeros(shape=(7, 256, 256), dtype=np.float)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            reference_image[id_7] += reference_single[i]
        return np.clip(reference_image, 0, 1)
    
    def _get_character_path(self, idx: str) -> str:
        """构建汉字图片路径"""
        # char_id = str(self.df.iloc[idx]['character'])
        char_id = idx
        return os.path.join(
            self.character_dir, 
            f"{char_id}{self.character_suffix}"
        )

    def _get_stroke_path(self, idx: str) -> str:
        """构建笔画图片路径"""
        #target = str(self.df.iloc[idx]['target'])
        target = idx
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
    
    
    def get_data(self, idx):
        """
        """
        
        char_id = self.valid_characters[idx]
        group = self.grouped_df.get_group(char_id)
        
        # 加载汉字图片
        char_path = self._get_character_path(char_id)
        char_img0= Image.open(char_path).convert('RGB')
        if self.character_transform:
            char_img = self.character_transform(char_img0)
            ref_img = self.reference_transform(char_img0)
        
        # 初始化笔画相关数据
        strokes = []
        ref_strokes = []
        stroke_orders = []
        targets = []
        stroke_labels = []  # 新建：真实笔画从1开始编号
        
       # 加载所有笔画
        for _, row in group.iterrows():
            stroke_path = self._get_stroke_path(row['target'])
            stroke_img0 = Image.open(stroke_path).convert('L')  # 转为灰度
            
            # 应用转换
            if self.stroke_transform:
                ref_stroke_img = self.reference_transform(stroke_img0)
                stroke_img = self.stroke_transform(stroke_img0)
                # ref_stroke_img = self.reference_transform(stroke_img)
            
            strokes.append(stroke_img)
            ref_strokes.append(ref_stroke_img)
            stroke_orders.append(row['stroke'] - 1)  # 转为0-based
            targets.append(row['target'])
            stroke_labels.append(row['stroke'])  # 真实笔画编号（1-based）
        
        # 将所有笔画堆叠成一个张量
        strokes_tensor = torch.cat(strokes,dim=0)
        ref_strokes_tensor = torch.cat(ref_strokes,dim=0)
        stroke_labels_tensor = torch.tensor(stroke_labels, dtype=torch.long)
        stroke_orders_tensor = torch.tensor(stroke_orders, dtype=torch.long)
    
        
        # reference_color = np.load(self.path[item][0])  # (3, 256, 256)
        # label_seg = np.load(self.path[item][1])[1:]  # (7, 256, 256)
        # target_image = np.load(self.path[item][1])[:1]  # (1, 256, 256)
        # seg_id = np.load(self.path[item][2])     # (N)
        # reference_transformed_single = np.load(self.path[item][3])  # (N, 256 256)
        # target_single_stroke = np.load(self.path[item][4])  # (N, 256 256)
        # target_data = np.repeat(target_image, 3, axis=0).astype(np.float)
        # reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        # label_seg = self.get_seg_image(target_single_stroke, seg_id)


        if self.is_single:  # For ExtractNet
            return {
                'target_data': char_img,
                'reference_color':ref_img,
                'label_seg': stroke_labels_tensor,
                'reference_segment_transformation_data':stroke_labels_tensor,
                'seg_id':stroke_labels,
                'reference_transformed_single': strokes_tensor,
                'target_single_stroke': strokes_tensor

            }
        else:  # For SegNet
            return {

                'target_data':  char_img,
                'reference_color': ref_img,
                'label_seg':stroke_labels_tensor
            }

    def __len__(self):
        return len(self.valid_characters)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data

