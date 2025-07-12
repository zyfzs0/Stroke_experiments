import os.path
import torch.utils.data as data
import pickle
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

'''
Load input data for SDNet
'''





class SDNetLoader_old(data.Dataset):
    def __init__(self, is_training, dataset_path, is_inference=False):
        self.is_inference = is_inference
        if is_training:
            self.path = [os.path.join(dataset_path, 'train', each) for each in os.listdir(os.path.join(dataset_path, 'train'))]
        else:
            self.path = [os.path.join(dataset_path, 'test', each) for each in os.listdir(os.path.join(dataset_path, 'test'))]
        print("number of dataset：%d" % len(self.path))

    def get_data(self, item):
        """
        """
        data_frame = np.load(self.path[item])
        reference_color_image = data_frame['reference_color_image']  # (3, 256, 256)
        reference_single_image = data_frame['reference_single_image']  # (N, 256 256)
        reference_single_centroid = data_frame['reference_single_centroid']
        target_image = data_frame['target_image']  # # (1, 256, 256)
        target_single_image = data_frame['target_single_image']  # (N, 256 256)
        stroke_label = data_frame['stroke_label']  # (N)

        stroke_num = reference_single_image.shape[0]
        expand_zeros = []
        expand_single_centroid = []

        for i in range(33):
            if i >= reference_single_image.shape[0]:
                expand_zeros.append(np.zeros(shape=(256, 256), dtype=float))
                expand_single_centroid.append(np.array([127.5, 127.5]))

        expand_zeros = np.array(expand_zeros)
        reference_single_image = np.concatenate([reference_single_image, expand_zeros], axis=0)
        target_single_image = np.concatenate([target_single_image, expand_zeros], axis=0)

        expand_single_centroid = np.array(expand_single_centroid)
        reference_single_centroid = np.concatenate([reference_single_centroid, expand_single_centroid], axis=0)

        if not self.is_inference:
            return {
                'target_single_stroke': target_single_image,
                'reference_single_stroke': reference_single_image,
                'target_data': target_image,
                'reference_color': reference_color_image,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': reference_single_centroid,
            }
        else:
            return {
                'target_single_stroke': target_single_image,
                'reference_single_stroke': reference_single_image,
                'target_data': target_image,
                'reference_color': reference_color_image,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': reference_single_centroid,
                'stroke_label': stroke_label,
            }




    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class SDNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_inference=False,
                 csv_path: str = None,
                 character_dir: str= None,
                 stroke_dir: str= None,
                 character_suffix: str = ".jpg",
                 stroke_suffix: str = ".jpg",
                 character_transform=None,
                 stroke_transform=None):
        self.is_inference = is_inference
        
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
        #     self.path = [os.path.join(dataset_path, 'train', each) for each in os.listdir(os.path.join(dataset_path, 'train'))]
        # else:
        #     self.path = [os.path.join(dataset_path, 'test', each) for each in os.listdir(os.path.join(dataset_path, 'test'))]
        # print("number of dataset：%d" % len(self.path))


    def _calculate_centroid(self, stroke_img):
        """计算笔画的质心坐标"""
        # 转换为numpy数组
        stroke_np = np.array(stroke_img.convert('L'))
        
        # 二值化 (假设笔画是黑色)
        _, binary = cv2.threshold(stroke_np, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 计算质心
        moments = cv2.moments(binary)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = 0, 0  # 如果没有笔画像素，返回(0,0)
        return torch.tensor([cx, cy], dtype=torch.float32)

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

    def __len__(self):
        return len(self.valid_characters)


    def get_data(self, idx):
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
        centroids = []
        stroke_orders = []
        targets = []
        stroke_labels = []  # 新建：真实笔画从1开始编号
        
       # 加载所有笔画
        for _, row in group.iterrows():
            stroke_path = self._get_stroke_path(row['target'])
            stroke_img0 = Image.open(stroke_path).convert('L')  # 转为灰度
            
            # 计算质心
            centroid = self._calculate_centroid(stroke_img0)
            
            # 应用转换
            if self.stroke_transform:
                ref_stroke_img = self.reference_transform(stroke_img0)
                stroke_img = self.stroke_transform(stroke_img0)
                # ref_stroke_img = self.reference_transform(stroke_img)
            
            strokes.append(stroke_img)
            ref_strokes.append(ref_stroke_img)
            centroids.append(centroid)
            stroke_orders.append(row['stroke'] - 1)  # 转为0-based
            targets.append(row['target'])
            stroke_labels.append(row['stroke']-1)  # 真实笔画编号（1-based）
        
        # 将所有笔画堆叠成一个张量
        strokes_tensor = torch.cat(strokes,dim=0)
        ref_strokes_tensor = torch.cat(ref_strokes,dim=0)
        centroids_tensor = torch.stack(centroids)
        stroke_orders_tensor = torch.tensor(stroke_orders, dtype=torch.long)
        stroke_labels_tensor = torch.tensor(stroke_labels, dtype=torch.long)
        # 填充到33个笔画
        max_strokes = 33
        num_strokes = len(strokes)
        num_real_strokes = len(strokes)
        
        if num_strokes < max_strokes:
            # 创建填充数据
            padding_images = torch.zeros(max_strokes - num_strokes, *strokes[0][0].shape)
            padding_refimages = torch.zeros(max_strokes - num_strokes, *ref_strokes[0][0].shape)
            padding_centroids = torch.full((max_strokes - num_strokes, 2), 127.5)
            padding_orders = torch.arange(num_strokes, max_strokes)  # 后续顺序编号
            padding_labels = torch.zeros(max_strokes - num_real_strokes, dtype=torch.long)  # 填充部分label=0
            # 填充
            ref_strokes_tensor = torch.cat([ref_strokes_tensor, padding_refimages],dim=0)
            strokes_tensor = torch.cat([strokes_tensor, padding_images],dim=0)
            centroids_tensor = torch.cat([centroids_tensor, padding_centroids],dim=0)
            stroke_orders_tensor = torch.cat([stroke_orders_tensor, padding_orders],dim=0)
            stroke_labels_tensor = torch.cat([stroke_labels_tensor, padding_labels],dim=0)
        elif num_strokes > max_strokes:
            ref_strokes_tensor = ref_strokes_tensor[:max_strokes]
            strokes_tensor = strokes_tensor[:max_strokes]
            centroids_tensor = centroids_tensor[:max_strokes]
            stroke_orders_tensor = stroke_orders_tensor[:max_strokes]
            stroke_labels_tensor = stroke_labels_tensor[:max_strokes]
        
        # print(strokes_tensor.shape,char_img.shape,ref_img.shape,centroids_tensor.shape,stroke_labels_tensor.shape)
        if not self.is_inference:
            return {
                'target_single_stroke': strokes_tensor,
                'reference_single_stroke': strokes_tensor,
                'target_data': char_img,
                'reference_color': ref_img,
                'stroke_num': num_strokes,
                'reference_single_stroke_centroid': centroids_tensor,
            }
        else:
            return {
                'target_single_stroke':strokes_tensor,
                'reference_single_stroke': strokes_tensor,
                'target_data': char_img,
                'reference_color': ref_img,
                'stroke_num': num_strokes,
                'reference_single_stroke_centroid': centroids_tensor,
                'stroke_label': stroke_labels_tensor,
            }
            
    def __getitem__(self, item):
        data = self.get_data(item)
        return data

            


if __name__ == '__main__':
    c = SDNetLoader(is_training=False, dataset_path='dataset/CCSEDB')
    data_loader = data.DataLoader(c, batch_size=8, shuffle=False)

    for i in range(10):
        print(i)
        for i_batch, sample_batched in enumerate(data_loader):

            style_single_image = sample_batched['style_single_image'][0].numpy()
            kaiti_single_image = sample_batched['kaiti_single_image'][0].numpy()
            original = sample_batched['original_style'].numpy()
            kaiti_color = sample_batched['kaiti_color'].numpy()
            stroke_num = sample_batched['stroke_num'][0].numpy()
            kaiti_center = sample_batched['kaiti_center'].numpy()
            assert (not np.isnan(kaiti_center).any())
            stroke_num = int(stroke_num)
            plt.figure(0)
            for j in range(8):
                plt.subplot2grid((2, 8), (0, j))
                plt.imshow(kaiti_color[j].transpose((1, 2, 0)))
                plt.subplot2grid((2, 8), (1, j))
                plt.imshow(original[j].squeeze())
            plt.show()
            # plt.figure(1)
            #
            # for j in range(stroke_num):
            #     plt.subplot2grid((2, stroke_num), (0, j))
            #     plt.imshow(kaiti_single_image[j])
            #     plt.subplot2grid((2, stroke_num), (1, j))
            #     plt.imshow(style_single_image[j])
            #
            # plt.show()

            # seg_id = sample_batched['seg_id'][0].numpy()
            #
            # data = { 'style_single_image': style_single_image,
            # 'kaiti_single_image': kaiti_single_image,
            # 'style_image': style_image,
            # 'kaiti_image': kaiti_image,
            # 'seg_id':seg_id}
            # save_dict(data, os.path.join(path, str(train_num)+'.pkl'))
            # train_num+=1
            # print(train_num)


