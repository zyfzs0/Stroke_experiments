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
from utils import seg_label_to7,seg_label_to7_o
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
        reference_image = np.zeros(shape=(7, 256, 256), dtype=float)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            # id_7 = seg_label_to7_o(seg_label[i])
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
        target_data = np.repeat(target_image, 3, axis=0).astype(float)
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


class SegNetExtractNetLoader_b(data.Dataset):
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
            csv_path = '/home/tongji209/majiawei/stroke_segmentation/split_data_by_character/train_metadata.csv'

        else:
            csv_path = '/home/tongji209/majiawei/stroke_segmentation/split_data_by_character/test_metadata.csv'
        
        
        self.df = pd.read_csv(csv_path)
        self.character_dir = '/home/tongji209/majiawei/stroke_segmentation/pixel_all_characters'
        self.stroke_dir = '/home/tongji209/majiawei/stroke_segmentation/pixel_all_strokes'
        
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
        # print("number of dataset：%d"%len(self.path)
    
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
    
    def get_seg_image(self, reference_single, seg_label):
        # print(reference_single.shape,seg_label.shape)
        reference_image = np.zeros(shape=(33, 256, 256), dtype=float)
        reference_image= torch.from_numpy(reference_image)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            # print(reference_image[id_7].shape,reference_single[i].shape)
            reference_image[id_7] = reference_image[id_7] + reference_single[i]
        return np.clip(reference_image, 0, 1)
    
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
            stroke_labels.append(row['stroke']-1)  # 真实笔画编号（1-based）
        

        
        n = len(strokes)
         # 转换为固定大小 (33, 256, 256)
        max_strokes = 33
        # 仅当 n > max_strokes 时进行截断
        if n > max_strokes:
            strokes = strokes[:max_strokes]
            ref_strokes = ref_strokes[:max_strokes]
            stroke_labels = stroke_labels[:max_strokes]
            stroke_orders = stroke_orders[:max_strokes]
        
        # 将所有笔画堆叠成一个张量
        strokes_tensor = torch.cat(strokes,dim=0)
        ref_strokes_tensor = torch.cat(ref_strokes,dim=0)
        stroke_labels_tensor = torch.tensor(stroke_labels, dtype=torch.long)
        stroke_orders_tensor = torch.tensor(stroke_orders, dtype=torch.long)
        
        # 转换为固定大小 (33, 256, 256)
        max_strokes = 33
        # final_strokes_tensor = torch.zeros(max_strokes, 256, 256, dtype=strokes_tensor.dtype)
        # final_strokes_tensor[:len(strokes)] = strokes_tensor  # 填充实际笔画
        final_strokes_tensor = strokes_tensor
        # reference_color = np.load(self.path[item][0])  # (3, 256, 256)
        # label_seg = np.load(self.path[item][1])[1:]  # (7, 256, 256)
        # target_image = np.load(self.path[item][1])[:1]  # (1, 256, 256)
        # seg_id = np.load(self.path[item][2])     # (N)
        # reference_transformed_single = np.load(self.path[item][3])  # (N, 256 256)
        # target_single_stroke = np.load(self.path[item][4])  # (N, 256 256)
        # target_data = np.repeat(target_image, 3, axis=0).astype(np.float)
        # reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        # label_seg = self.get_seg_image(target_single_stroke, seg_id)

        # print(ref_img.shape,stroke_labels_tensor.shape,strokes_tensor.shape,stroke_labels)

        
        reference_segment_transformation_data = self.get_seg_image(final_strokes_tensor, stroke_labels_tensor)
        zeros_tensor_reference_seg = torch.zeros_like(reference_segment_transformation_data)
        zero_transformed_single = torch.zeros_like(strokes_tensor)
        label_seg = self.get_seg_image(final_strokes_tensor, stroke_labels_tensor)
        if self.is_single:  # For ExtractNet
            return {
                'target_data': ref_img,
                'reference_color':ref_img,
                'label_seg': label_seg,
                'reference_segment_transformation_data':zeros_tensor_reference_seg,
                'seg_id':stroke_labels_tensor,
                'reference_transformed_single':zero_transformed_single,    #strokes_tensor,
                'target_single_stroke': strokes_tensor

            }
        else:  # For SegNet
            return {

                'target_data':  ref_img,
                'reference_color': ref_img,
                'label_seg':label_seg
            }

    def __len__(self):
        return len(self.valid_characters)

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
            csv_path = '/home/tongji209/majiawei/stroke_segmentation/split_data_by_character/train_metadata.csv'

        else:
            csv_path = '/home/tongji209/majiawei/stroke_segmentation/split_data_by_character/test_metadata.csv'
        
        self.rd_df = pd.read_csv('/home/tongji209/majiawei/Stroke_experiments/RHSEDB/character_info.csv')
        self.rd_path = '/home/tongji209/majiawei/Stroke_experiments/RHSEDB/'
        
        self.df = pd.read_csv(csv_path)
        self.character_dir = '/home/tongji209/majiawei/stroke_segmentation/pixel_all_characters'
        self.stroke_dir = '/home/tongji209/majiawei/stroke_segmentation/pixel_all_strokes'
        
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
            rd_mes = self._rd_get_char(char_id)
            if not rd_mes :
                valid = False
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
        # print("number of dataset：%d"%len(self.path)
    
    def _rd_get_char(self, char_id):
        unicode_value = self.rd_df.loc[self.rd_df['unicode'] == char_id]
        if not unicode_value.empty:
            num_strokes = unicode_value['stroke_count'].iloc[0]
            dataset = unicode_value['dataset'].iloc[0]
            npz_name = unicode_value['npz_name'].iloc[0]
            path = os.path.join(self.rd_path, dataset, npz_name)
            return {'num': num_strokes, 'path': path}
        else:
            return None
        return None
    
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
    
    def get_seg_image(self, reference_single, seg_label):
        # print(reference_single.shape,seg_label.shape)
        # reference_image = np.zeros(shape=(33, 256, 256), dtype=float)
        reference_image = np.zeros(shape=(7, 256, 256), dtype=float)
        reference_image= torch.from_numpy(reference_image)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7_o(seg_label[i])
            # print(reference_image[id_7].shape,reference_single[i].shape)
            reference_image[id_7] = reference_image[id_7] + reference_single[i]
        return np.clip(reference_image, 0, 1)
    
    def get_data(self, idx):
        """
        """
        item = idx
        char_id = self.valid_characters[idx]
        group = self.grouped_df.get_group(char_id)
        rd_mes = self._rd_get_char(char_id)
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
            stroke_labels.append(row['stroke']-1)  # 真实笔画编号（1-based）
        kaiti_ref = False
        if rd_mes:
            data_frame = np.load(rd_mes['path'])
            if kaiti_ref:
                reference_color_image = data_frame['reference_color_image']  # (3, 256, 256)
                reference_single_image = data_frame['reference_single_image']  # (N, 256 256)
            else:
                reference_color_image = data_frame['target_image']  # (1, 256, 256
                reference_color_image = np.repeat(reference_color_image, 3, axis=0)
                reference_single_image = data_frame['target_single_image']  # (N, 256 256
            n_strokes_o = len(strokes)
            n_strokes = min(n_strokes_o,rd_mes['num'])
            reference_single_image = reference_single_image[:n_strokes]
            # print(reference_single_image.shape,n_strokes)
            reference_color_image = torch.from_numpy(reference_color_image).float()
            reference_single_image = torch.from_numpy(reference_single_image).float()
        else:
            reference_color_image = ref_img # (3, 256, 256)
            reference_single_image = torch.cat(strokes,dim=0)
            n_strokes = len(strokes)
        
        # centroids_tensor = centroids_tensor[:n_strokes]
        stroke_orders= stroke_orders[:n_strokes]
        stroke_label= stroke_labels[:n_strokes]
        # 填充到33个笔画
        max_strokes = 33
        num_strokes = n_strokes
        num_real_strokes = n_strokes
        
        n = n_strokes
         # 转换为固定大小 (33, 256, 256)
        max_strokes = 33
        # 仅当 n > max_strokes 时进行截断
        if n > max_strokes:
            strokes = strokes[:max_strokes]
            reference_single_image = reference_single_image [:max_strokes]
            ref_strokes = ref_strokes[:max_strokes]
            stroke_labels = stroke_labels[:max_strokes]
            stroke_orders = stroke_orders[:max_strokes]
        
        # 将所有笔画堆叠成一个张量
        strokes_tensor = torch.cat(strokes,dim=0)
        stroke_labels_tensor = torch.tensor(stroke_labels, dtype=torch.long)
        stroke_orders_tensor = torch.tensor(stroke_orders, dtype=torch.long)
        
        # 转换为固定大小 (33, 256, 256)
        max_strokes = 33
        
        if n<max_strokes:
            stroke_fill = stroke_fill = torch.zeros((max_strokes - n,) + strokes_tensor.shape[1:])
            all_strokes = torch.cat([strokes_tensor, stroke_fill], dim=0)
        else:
            all_strokes = strokes_tensor
        # final_strokes_tensor = torch.zeros(max_strokes, 256, 256, dtype=strokes_tensor.dtype)
        # final_strokes_tensor[:len(strokes)] = strokes_tensor  # 填充实际笔画
        final_strokes_tensor = strokes_tensor
        reference_color = np.load(self.path[item][0])  # (3, 256, 256)
        label_seg = np.load(self.path[item][1])[1:]  # (7, 256, 256)
        target_image = np.load(self.path[item][1])[:1]  # (1, 256, 256)
        seg_id = np.load(self.path[item][2])     # (N)
        reference_transformed_single = np.load(self.path[item][3])  # (N, 256 256)
        num_real_strokes = reference_transformed_single.shape[0]
        # print(num_real_strokes)
        target_single_stroke = np.load(self.path[item][4])  # (N, 256 256
        current_channels = target_single_stroke.shape[0]
        if current_channels < 33:
            # 计算需要补充的零张量数量
            pad_size = 33 - current_channels
            # 在第一个维度（通道维度）补零
            padded = np.pad(target_single_stroke, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            target_single_stroke_all= padded
        elif current_channels > 33:
            # 如果通道数超过33，可以截断或报错
            target_single_stroke_all = target_single_stroke[:33]  # 取前33个通道
        target_data = np.repeat(target_image, 3, axis=0).astype(float)
        reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        label_seg = self.get_seg_image(target_single_stroke, seg_id)

        # print(ref_img.shape,stroke_labels_tensor.shape,strokes_tensor.shape,stroke_labels)

        # print(target_single_stroke.shape)
        # reference_segment_transformation_data = self.get_seg_image(reference_single_image, stroke_labels_tensor)
        # zeros_tensor_reference_seg = torch.zeros_like(reference_segment_transformation_data)
        # zero_transformed_single = torch.zeros_like(strokes_tensor)
        # label_seg = self.get_seg_image(final_strokes_tensor, stroke_labels_tensor)
        # if self.is_single:  # For ExtractNet
        #     return {
        #         'target_data': ref_img,
        #         'reference_color':reference_color_image,
        #         'label_seg': label_seg,
        #         'reference_segment_transformation_data': reference_segment_transformation_data,
        #         'seg_id':stroke_labels_tensor,
        #         'reference_transformed_single': reference_single_image,    #strokes_tensor,
        #         'target_single_stroke': strokes_tensor,
        #         'stroke_num':num_real_strokes,

        #     }
        # else:  # For SegNet
        #     return {

        #         'target_data':  ref_img,
        #         'reference_color': reference_color_image,
        #         'label_seg':label_seg,
        #         'target_single_stroke':all_strokes,
        #         'stroke_num':num_real_strokes,
        #     }

        
        if self.is_single:  # For ExtractNet
            return {
                'target_data': target_data,
                'reference_color':reference_color,
                'label_seg': label_seg,
                'reference_segment_transformation_data':reference_segment_transformation_data,
                'seg_id': seg_id,
                'reference_transformed_single': reference_transformed_single,
                'target_single_stroke': target_single_stroke,
                'stroke_num':num_real_strokes,
                'unicode':char_id ,

            }
        else:  # For SegNet
            return {

                'target_data': target_data,
                'reference_color': reference_color,
                'label_seg': label_seg,
                'target_single_stroke':target_single_stroke_all,
                'stroke_num':num_real_strokes,
            }
            
    def __len__(self):
        return len(self.valid_characters)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data