import torch
import torch.nn.functional as F
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.optim as optim
import torch.utils.data as data
from model.model_of_ExtractNet import ExtractNet
from model.model_of_SegNet import SegNet
from load_data_for_SegNetExtractNet import SegNetExtractNetLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from utils import random_colors, apply_stroke_t, save_picture, seg_colors, apply_stroke, seg_label_to7
from utils_loss_val import get_iou_without_matching, get_iou_with_matching
import piq
from piq import psnr, ssim
import lpips
from CLDICE.cldice import soft_dice,soft_cldice,soft_dice_cldice

device = torch.device("cuda:0")
device0 = torch.device('cpu')
loss_fn_lpips = lpips.LPIPS(net='alex').to(device0)  # 使用 AlexNet 作为 LPIPS 的特征提取器

def min_max_normalize(x, target_data_o):
    """
    针对黑色有效/白色无效的target_data_o的特殊处理
    返回归一化后的(3,256,256)数组，其中：
    - 有效区域保持原始黑色特征
    - 无效区域设为白色
    
    参数:
        x: 二值掩码 (True表示有效区域)
        target_data_o: 参考图像 (黑色有效，白色无效)
    """
    # 转换数据类型
    mask = x.astype(np.float32)  # [H,W]
    target_np = target_data_o.detach().cpu().numpy()  # [3,H,W]
    
    # 创建白色背景 (值设为1.0)
    white_bg = np.ones_like(target_np)  # 全1表示白色
    
    # 应用掩码：有效区域用原图，无效区域用白色
    # 注意：这里需要广播mask到3个通道
    masked = target_np * (1 - mask) + white_bg * mask
    
    # 归一化处理
    c_min = masked.min()
    c_max = masked.max()
    normalized = (masked - c_min) / (c_max - c_min + 1e-8)
    # 假设 normalized 是形状为 (3,256,256) 的 numpy 数
    # print(normalized.shape)
    # normalized_uint8 = (normalized[0] * 255.0).clip(0, 255).astype(np.uint8) # 转换为 [0,255] 的 uint8

    # # 调整通道顺序为 (256,256,3) 供 matplotlib 显示
    # normalized_rgb = np.transpose(normalized_uint8, (1, 2, 0))
    # normalized_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./test_pic/normalized_image_cv.png', normalized_bgr)
    return normalized.astype(np.float32)


def compute_iou_batch(pred, target, eps=1e-8):
    """
    逐图计算 IoU，返回 [B] IoU 列表
    """
    B = pred.shape[0]
    intersection = ((pred == 1) & (target == 1)).float().view(B, -1).sum(dim=1)
    union = ((pred == 1) | (target == 1)).float().view(B, -1).sum(dim=1)
    return intersection / (union + eps)  # shape: [B]


def Dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1e-5
    intersection = np.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return (1. - coeff)

def compute_iou(pred, target, eps=1e-8):
    """    
    返回:
        mean_iou: 批量的平均IoU
    """
    # 统一为4维张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 确保是三通道（如果是单通道则复制）
    if pred.size(1) == 1:
        pred = pred.repeat(1, 3, 1, 1)
    if target.size(1) == 1:
        target = target.repeat(1, 3, 1, 1)
    
    # 初始化存储各通道IoU
    channel_ious = []
    
    # 对每个通道计算IoU
    for c in range(3):
        # 获取当前通道
        pred_c = pred[:, c]  # [n,h,w]
        target_c = target[:, c]  # [n,h,w]
        
        # 二值化（如果pred是概率图）
        pred_bin = (pred_c < 0.5).float()
        target_bin = (target_c < 0.5).float()
        
        # 计算IoU
        intersection = (pred_bin * target_bin).sum(dim=[1,2])  # [n]
        union = ((pred_bin + target_bin) > 0).float().sum(dim=[1,2])  # [n]
        
        iou = (intersection + eps) / (union + eps)  # [n]
        channel_ious.append(iou)
    
    # 计算平均IoU（先对各样本取通道平均，再对batch取平均）
    mean_iou = torch.stack(channel_ious).mean(dim=0).mean()  # 先平均通道，再平均batch
    
    return mean_iou


def batch_soft_dice(y_true, y_pred):
    B = y_true.shape[0]
    losses=[]
    for b in range(B):
        yt = 1.0 - y_true[b]
        yp = 1.0 - y_pred[b]
        losses.append(soft_dice(yt, yp))
    return sum(losses)/B




def batch_mask_dice(y_true, y_pred):
    B = len(y_true)
    losses=[]
    # mask_losses = []
    for b in range(B):
        yt = 1.0 - y_true[b]
        yp = 1.0 - y_pred[b]
        losses.append(Dice(yt, yp))
    return sum(losses)/B

def compute_pixel_accuracy_batch(pred, target):
    """
    逐图计算 Pixel Accuracy，返回 [B] 精度列表
    """
    B = pred.shape[0]
    correct = (pred == target).float().view(B, -1).sum(dim=1)
    total = pred[0].numel()
    return correct.sum() / (B * total)  # 标量平均值


def batch_normalize_to_01(x, eps=1e-8):
    # x: [B, C, H, W]
    x_flat = x.view( -1)
    min_vals = x_flat.min(dim=1, keepdim=True)[0]
    max_vals = x_flat.max(dim=1, keepdim=True)[0]
    x_norm = (x_flat - min_vals) / (max_vals - min_vals + eps)
    return x_norm.view_as(x)

def batch_normalize_to_minus1_1(x: torch.Tensor, eps=1e-8):
    """
    对 [B,C,H,W] 的图像进行 per-sample 归一化到 [-1,1]
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)
    x_min = x_flat.min(dim=1, keepdim=True)[0]
    x_max = x_flat.max(dim=1, keepdim=True)[0]
    x_norm = (x_flat - x_min) / (x_max - x_min + eps)  # -> [0,1]
    x_norm = x_norm * 2 - 1  # -> [-1,1]
    return x_norm.view_as(x)


class DataPool(object):
    '''
    In actual situations, due to the different number of strokes in each character, in order to achieve batch training,
    we build a data pool
    '''
    def __init__(self):
        self.data_num = 0
        self.pool_data = {}

    def add(self, target_data, reference_stroke_transformation_data, segment_data,
            reference_segment_transformation_data, segNet_feature, label, cut_box_list):
        '''
        put data into pool
        '''

        if 'target_data' not in self.pool_data.keys():
            self.pool_data['target_data'] = target_data
        else:
            self.pool_data['target_data'] = torch.cat([self.pool_data['target_data'], target_data],
                                                           dim=0)

        if 'reference_stroke_transformation_data' not in self.pool_data.keys():
            self.pool_data['reference_stroke_transformation_data'] = reference_stroke_transformation_data
        else:
            self.pool_data['reference_stroke_transformation_data'] = torch.cat([self.pool_data['reference_stroke_transformation_data'], reference_stroke_transformation_data], dim=0)

        if 'segment_data' not in self.pool_data.keys():
            self.pool_data['segment_data'] = segment_data
        else:
            self.pool_data['segment_data'] = torch.cat(
                [self.pool_data['segment_data'], segment_data], dim=0)

        if 'reference_segment_transformation_data' not in self.pool_data.keys():
            self.pool_data['reference_segment_transformation_data'] = reference_segment_transformation_data
        else:
            self.pool_data['reference_segment_transformation_data'] = torch.cat([self.pool_data['reference_segment_transformation_data'], reference_segment_transformation_data],
                                                              dim=0)

        if 'segNet_feature' not in self.pool_data.keys():
            self.pool_data['segNet_feature'] = segNet_feature
        else:
            self.pool_data['segNet_feature'] = torch.cat([self.pool_data['segNet_feature'], segNet_feature], dim=0)

        if 'label' not in self.pool_data.keys():
            self.pool_data['label'] = label
        else:
            self.pool_data['label'] = torch.cat([self.pool_data['label'], label], dim=0)

        if 'cut_box_list' not in self.pool_data.keys():
            self.pool_data['cut_box_list'] = cut_box_list
        else:
            self.pool_data['cut_box_list'].extend(cut_box_list)

        self.data_num += int(label.size(0))

    def next(self, num):
        # get next training data from pool

        target_data_batch = self.pool_data['target_data'][:num]
        self.pool_data['target_data'] = self.pool_data['target_data'][num:]

        reference_stroke_transformation_data_batch = self.pool_data['reference_stroke_transformation_data'][:num]
        self.pool_data['reference_stroke_transformation_data'] = self.pool_data['reference_stroke_transformation_data'][num:]

        segment_data_batch = self.pool_data['segment_data'][:num]
        self.pool_data['segment_data'] = self.pool_data['segment_data'][num:]

        reference_segment_transformation_data_batch = self.pool_data['reference_segment_transformation_data'][:num]
        self.pool_data['reference_segment_transformation_data'] = self.pool_data['reference_segment_transformation_data'][num:]

        segNet_feature_batch = self.pool_data['segNet_feature'][:num]
        self.pool_data['segNet_feature'] = self.pool_data['segNet_feature'][num:]

        label_batch = self.pool_data['label'][:num]
        self.pool_data['label'] = self.pool_data['label'][num:]

        cut_list_batch = self.pool_data['cut_box_list'][:num]
        self.pool_data['cut_box_list'] = self.pool_data['cut_box_list'][num:]

        self.data_num -= num

        return [target_data_batch, reference_stroke_transformation_data_batch, segment_data_batch,
        reference_segment_transformation_data_batch, segNet_feature_batch, label_batch, cut_list_batch]





class TrainExtractNet():
    '''
        train SDNet with the Train-Dataset
        validate SDNet with the Test-Dataset
    '''

    def __init__(self,  save_path=None, segNet_save_path=None):
        super().__init__()
        self.segNet_save_path = segNet_save_path
        self.Out_path_train = os.path.join(save_path, 'train')
        self.Model_path = os.path.join(save_path, 'model')
        self.Out_path_loss = os.path.join(save_path, 'loss')
        self.Out_path_val = os.path.join(save_path, 'val')
        if not os.path.exists(self.Model_path):
            os.makedirs(self.Model_path)
        if not os.path.exists(self.Out_path_train):
            os.makedirs(self.Out_path_train)
        if not os.path.exists(self.Out_path_loss):
            os.makedirs(self.Out_path_loss)
        if not os.path.exists(self.Out_path_val):
            os.makedirs(self.Out_path_val)

        # SegNet
        self.seg_net = SegNet(out_feature=True)


        # Extract
        self.extract_net = ExtractNet()
        self.extract_net.to(device)

        # Data Pool
        self.data_pool = DataPool()

    def save_model_parameter(self, epoch):
        # save models
        state_stn = {'net': self.extract_net.state_dict(), 'start_epoch': epoch}
        torch.save(state_stn, os.path.join(self.Model_path, 'model_extract.pth'))

    def train_model(self, epochs=40, batch_size=16, init_learning_rate=0.001, dataset=None):
        self.batch_size = batch_size

        # load parameters of SegNet
        seg_model_path = os.path.join(self.segNet_save_path, 'model', 'model.pth')
        state = torch.load(seg_model_path)
        self.seg_net.load_state_dict(state['net'])
        self.seg_net.to(device).eval().requires_grad_(False)

        # dataset
        train_loader = data.DataLoader(SegNetExtractNetLoader(is_training=True, dataset_path=dataset, is_single=True), batch_size=1, shuffle=True)
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset, is_single=True), batch_size=1)
        optim_op = optim.Adam(self.extract_net.parameters(), lr=init_learning_rate, betas=(0.5, 0.999))
        lr_scheduler_op = optim.lr_scheduler.ExponentialLR(optim_op, gamma=0.5)

        train_history_loss = []
        test_history_loss = []

        for i in range(epochs):
            print("Start training the %d epoch" % (i + 1))
            train_loss, loss_name = self.__train_epoch(i, train_loader, optim_op)
            test_loss, loss_name = self.__val_epoch(i, test_loader)
            train_history_loss.append(train_loss)
            test_history_loss.append(test_loss)
            for index, name in enumerate(loss_name):
                train_data = [x[index] for x in train_history_loss]
                test_data = [x[index] for x in test_history_loss]
                self.__plot_loss(name+'stage2.png', [train_data, test_data],
                               legend=['train', 'test'], folder_name=self.Out_path_loss)
            # save models
            self.save_model_parameter(i)
            if (i+1)%5 == 0:
                lr_scheduler_op.step()

    def test_model(self, extract_save_path =None,dataset=None):

        # load parameters of SegNet
        seg_model_path = os.path.join(self.segNet_save_path, 'model', 'model.pth')
        state = torch.load(seg_model_path)
        self.seg_net.load_state_dict(state['net'])
        self.seg_net.to(device).eval().requires_grad_(False)

        extract_model_path = os.path.join(extract_save_path, 'model', 'model_extract.pth')
        ex_state = torch.load(extract_model_path)
        self.extract_net.load_state_dict(ex_state['net'])
        self.extract_net.to(device).eval().requires_grad_(False)
        # dataset
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset, is_single=True), batch_size=1)
        test_history_loss = []
        test_loss, loss_name = self.__val_epoch(20, test_loader)
        test_history_loss.append(test_loss)
        for index, name in enumerate(loss_name):
            test_data = [x[index] for x in test_history_loss]
            self.__plot_loss(name+'stage2.png', [test_data],
                            legend=['test'], folder_name=self.Out_path_loss)
            
    def __plot_loss(self, name, loss, legend, folder_name, save=True):
        '''
        @param name: name
        @param loss: array,shape=(N, 2)
        @return:
        '''
        loss_ = np.array(loss)
        plt.figure("loss")
        plt.gcf().clear()

        for i in range(len(legend)):
            plt.plot(loss_[i,:], label=legend[i])

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if save:
            save_path = os.path.join(folder_name, name)
            plt.savefig(save_path)
        else:
            plt.show()

    def __get_cut_region(self, kaiti_imagae):
        if np.sum(kaiti_imagae) > 5:
            points = np.where(kaiti_imagae > 0.5)
            x_l = np.min(points[1])
            x_r = np.max(points[1])
            y_t = np.min(points[0])
            y_b = np.max(points[0])
        else:
            y_t = 0
            y_b = 255
            x_l = 0
            x_r = 255
        w = x_r - x_l + 1
        h = y_b - y_t + 1
        center_x = int((x_l+x_r)/2)
        center_y = int((y_t+y_b)/2)
        size = max(w, h)
        if size>32:
            cut_size = 256
            # y_t, y_b, x_l, x_r
            return [0, 256, 0, 256]
        elif size>16:
            cut_size = 128
        else:
            cut_size = 64

        if center_x - cut_size/2 <= 0:
            x_l = 0
            x_r = cut_size
        elif center_x+cut_size/2>=256:
            x_l = 256-cut_size
            x_r = 256
        else:
            x_l = int(center_x - cut_size/2)
            x_r = x_l + cut_size

        if center_y - cut_size/2 <= 0:
            y_t = 0
            y_b = cut_size
        elif center_y+cut_size/2>=256:
            y_t = 256-cut_size
            y_b = 256
        else:
            y_t = int(center_y - cut_size/2)
            y_b = y_t + cut_size

        return [y_t, y_b, x_l, x_r]

    def __create_color_image(self, image, id):
        image = image.repeat(1, 3, 1, 1)
        color = torch.from_numpy(np.array(seg_colors[id]).reshape(1, 3, 1, 1)).to(device)
        return image * color

    def __get_training_data_of_ExtarctNet(self, reference_transformed_single, target_single_stroke, seg_index,
                                   seg_out, reference_segment_transformation_data,
                                          target_data, seg_out_feature):
        '''
        get training_data of ExtractNet

        When the size of reference stroke is too small, the stroke area is clipped and enlarged
        to increase the discrimination ability.
        '''

        kaiti_tran_single_image = torch.reshape(reference_transformed_single, shape=(-1, 1, 256, 256))
        style_single_image = torch.reshape(target_single_stroke, shape=(-1, 1, 256, 256))
        kaiti_trans_stage2_in = []
        kaiti_trans_seg_stage2_in = []
        style_stage2_in = []
        seg_out_stage_in = []
        style_original_stage2_in = []
        cut_box_list = []
        for index in range(int(seg_index.size(0))):
            id = int(seg_index[index])
            id_7 = seg_label_to7(id)
            kaiti_single_image_trans = kaiti_tran_single_image[index:index + 1]

            # get cut region：
            kaiti_image__ = np.squeeze(kaiti_single_image_trans.to('cpu').numpy())
            cut_box = self.__get_cut_region(kaiti_image__)
            torch_resize = Resize([256, 256])
            y_t, y_b, x_l, x_r = cut_box
            cut_box_list.append(cut_box)

            # Reference Stroke Transformation Data
            kaiti_tran_in = kaiti_single_image_trans.detach()
            kaiti_tran_in = torch_resize(kaiti_tran_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_tran_in = self.__create_color_image(kaiti_tran_in, id).float()
            kaiti_trans_stage2_in.append(kaiti_tran_in)

            # label
            style_in = torch.unsqueeze(style_single_image[index], dim=0).float()
            style_in = torch_resize(style_in[:, :, y_t:y_b, x_l:x_r])
            style_stage2_in.append(style_in)

            # Segment Data
            seg_out_in = F.sigmoid(seg_out[:, id_7:id_7 + 1]).detach().float()
            seg_out_in = torch_resize(seg_out_in[:, :, y_t:y_b, x_l:x_r])
            seg_out_stage_in.append(seg_out_in)

            # Reference Segment Transformation Data
            kaiti_seg_in = reference_segment_transformation_data[:, id_7:id_7 + 1].detach().float()
            # print("kaiti_seg_in shape:", kaiti_seg_in.shape) 
            kaiti_seg_in = torch_resize(kaiti_seg_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_trans_seg_stage2_in.append(kaiti_seg_in)

            # Target Data
            style_image_original_in = torch_resize(target_data[:, :, y_t:y_b, x_l:x_r])
            style_original_stage2_in.append(style_image_original_in)
        reference_stroke_transformation_data = torch.cat(kaiti_trans_stage2_in, dim=0)
        label = torch.cat(style_stage2_in, dim=0)
        segment_data = torch.cat(seg_out_stage_in, dim=0)
        segNet_feature = seg_out_feature['out_64_32'].repeat(label.size(0), 1, 1, 1)
        target_data = torch.cat(style_original_stage2_in, dim=0)
        reference_segment_transformation_data = torch.cat(kaiti_trans_seg_stage2_in, dim=0)

        return [target_data, reference_stroke_transformation_data, segment_data,
                reference_segment_transformation_data, segNet_feature, label, cut_box_list]

    def __to_original_stroke(self, out, label, cut_box_list):
        '''
        Restore strokes to their original size
        :param cut_box_list: Adaptive size parameters
        :return:
        '''
        out = out.squeeze(1).to('cpu').numpy()
        label = label.squeeze(1).to('cpu').numpy()
        out_re = []
        label_re = []

        for i in range(len(cut_box_list)):
            y_t, y_b, x_l, x_r = cut_box_list[i]
            img = np.zeros_like(out[i])
            img[y_t:y_b, x_l:x_r] = cv2.resize(out[i].astype(np.uint8), dsize=(x_r - x_l, y_b - y_t))
            out_re.append(img)

            img_r = np.zeros_like(label[i])
            img_r[y_t:y_b, x_l:x_r] = cv2.resize(label[i].astype(np.uint8), dsize=(x_r - x_l, y_b - y_t))
            label_re.append(img_r)
        return out_re, label_re

    def __train_epoch(self, epoch, train_loader, optim_opWhole):
        epoch += 1
        self.extract_net.train()
        loss_list = []
        start_time = time.time()

        pool_i = 0

        for i, batch_sample in enumerate(train_loader):
            # get data
            reference_color = batch_sample['reference_color'].float().to(device)
            reference_segment_transformation_data = batch_sample['reference_segment_transformation_data'].float().to(device)
            target_data = batch_sample['target_data'].float().to(device)
            target_single_stroke = batch_sample['target_single_stroke'].float().to(device)
            reference_transformed_single = batch_sample['reference_transformed_single'].float().to(device)
            seg_index = batch_sample['seg_id'][0].long().to(device)
            target_data_o = target_data.clone()
            # get segment result fo SegNet
            seg_out, seg_out_feature = self.seg_net(target_data, reference_color)

            # get inputs of ExtractNet
            target_data, reference_stroke_transformation_data, segment_data, \
            reference_segment_transformation_data, segNet_feature, label, cut_box_list = self.__get_training_data_of_ExtarctNet(reference_transformed_single, target_single_stroke, seg_index,
                                   seg_out, reference_segment_transformation_data,
                                          target_data, seg_out_feature)
            # put data into pool
            self.data_pool.add(target_data, reference_stroke_transformation_data, segment_data,
                                reference_segment_transformation_data, segNet_feature, label, cut_box_list)

            while self.data_pool.data_num >= self.batch_size:
                target_data_batch, reference_stroke_transformation_data_batch, segment_data_batch,\
                reference_segment_transformation_data_batch, segNet_feature_batch, label_batch, cut_box_list_batch = self.data_pool.next(self.batch_size)

                extract_out = self.extract_net(reference_stroke_transformation_data_batch,
                                                       reference_segment_transformation_data_batch, segment_data_batch,
                                                       target_data_batch, segNet_feature_batch)

                # calculate loss
                loss = F.binary_cross_entropy(F.sigmoid(extract_out), label_batch)
                extract_result = F.sigmoid(extract_out).detach() > 0.5
                
                #  Restore strokes to their original size
                # Calculate mIOUm
                #  Restore strokes to their original size
                extract_result, label = self.__to_original_stroke(extract_result, label_batch, cut_box_list_batch)
                mIOUm = get_iou_with_matching(extract_result, label)

                # mIOUum is not need to be calculated in the Training
                extract_result_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in extract_result]
                label_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in label]
                # print(extract_result_norm[0].shape)
                # 转换为PIQ需要的格式 (N, 1, H, W)
                extract_tensor = torch.cat([x for x in extract_result_norm], dim=0)  # 沿批次维拼接
                label_tensor = torch.cat([x for x in label_norm], dim=0)

                #print(extract_tensor.shape)
                # 计算指标
                mse = torch.nn.functional.mse_loss(extract_tensor, label_tensor)
                psnr_value = psnr(extract_tensor, label_tensor, data_range=1.0)
                ssim_value = ssim(extract_tensor, label_tensor, data_range=1.0)
                iou_value = compute_iou(extract_tensor, label_tensor)
                dice_value = batch_soft_dice(label_tensor, extract_tensor)
                dice_mask_value = batch_mask_dice(label,extract_result)
                accuarcy_value = compute_pixel_accuracy_batch(extract_tensor, label_tensor)
                # LPIPS需要3通道
                extract_rgb = extract_tensor.repeat(1, 1, 1, 1)
                label_rgb = label_tensor.repeat(1, 1, 1, 1)
                
                # 将数据从 [0,1] 转换到 [-1,1]（LPIPS 要求）
                extract_tensor_lpips = extract_tensor * 2 - 1  # [0,1] -> [-1,1]
                label_tensor_lpips = label_tensor * 2 - 1      # [0,1] -> [-1,1]

                # 扩展为 3 通道（LPIPS 需要 RGB）
                extract_rgb = extract_tensor_lpips.repeat(1, 1, 1, 1)  # (N, 3, H, W)
                label_rgb = label_tensor_lpips.repeat(1, 1, 1, 1)      # (N, 3, H, W)

                # 计算 LPIPS
                lpips_value = loss_fn_lpips(extract_rgb, label_rgb).mean()

                loss.backward()
                optim_opWhole.step()
                optim_opWhole.zero_grad()

                torch.cuda.empty_cache()
                loss_list.append([
                    loss.item(), 
                    mIOUm.item(), 
                    0.0,
                    mse.item(),       # MSE (越小越好)
                    psnr_value.item(), # PSNR (越大越好)
                    ssim_value.item(), # SSIM (越大越好)
                    lpips_value.item(),# LPIPS (越小越好)
                    iou_value.item(),
                    dice_value.item(),
                    dice_mask_value.item(),
                    accuarcy_value.item(),
                ])




        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['loss', 'mIOUm', 'mIOUum', 'MSE', 'PSNR', 'SSIM', 'LPIPS','IOU','Dice','Dicem','Accuracy']
        print(
            "[TRAIN][{}/{}], loss={:.7f}, mIOUm={:.7f}, mIOUum={:.7f}, "
            "MSE={:.7f}, PSNR={:.7f}, SSIM={:.7f}, LPIPS={:.7f},IOU={:.7f},Dice={:.7f},Dicem={:.7f}, Accuracy={:.7f},time={:.7f}".format(
                i, len(train_loader), 
                loss_value[0], loss_value[1], loss_value[2],
                loss_value[3], loss_value[4], loss_value[5], loss_value[6],loss_value[7],loss_value[8],loss_value[9],loss_value[10],
                time.time() - start_time
            )
        )
        return loss_value, loss_name

    def __val_epoch(self, epoch, test_loader):
        epoch += 1
        self.extract_net.eval()
        loss_list = []
        start_time = time.time()

        for i, batch_sample in enumerate(test_loader):
            # get data
            reference_color = batch_sample['reference_color'].float().to(device)
            reference_segment_transformation_data = batch_sample['reference_segment_transformation_data'].float().to(device)
            target_data_o = batch_sample['target_data'].float().to(device)
            target_single_stroke = batch_sample['target_single_stroke'].float().to(device)
            reference_transformed_single = batch_sample['reference_transformed_single'].float().to(device)
            seg_index = batch_sample['seg_id'][0].long().to(device)
            srtrokes_num =batch_sample['stroke_num'].to(device0)
            # get segment result fo SegNet
            seg_out, seg_out_feature = self.seg_net(target_data_o, reference_color)

            # get inputs of ExtractNet
            target_data, reference_stroke_transformation_data, segment_data, \
            reference_segment_transformation_data, segNet_feature, label, cut_box_list = self.__get_training_data_of_ExtarctNet(
                reference_transformed_single, target_single_stroke, seg_index,
                seg_out, reference_segment_transformation_data,
                target_data_o, seg_out_feature)

            extract_out = self.extract_net(reference_stroke_transformation_data,
                                           reference_segment_transformation_data, segment_data,
                                           target_data, segNet_feature)
            # print(len(label),len(extract_out))
            # calculate loss
            loss = F.binary_cross_entropy(F.sigmoid(extract_out), label)
            extract_result = F.sigmoid(extract_out).detach() > 0.5

            #  Restore strokes to their original size
            extract_result, label = self.__to_original_stroke(extract_result, label, cut_box_list)
            # print(extract_result[0].shape,label[0].shape) # (256,256)
            # Calculate mIOUm and mIOUum
            mIOUm = get_iou_with_matching(extract_result, label)
            mIOUum = get_iou_without_matching(extract_result, label)
            
            extract_result_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in extract_result]
            label_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in label]

            # 转换为PIQ需要的格式 (N, 1, H, W)
            extract_tensor = torch.cat([x for x in extract_result_norm], dim=0)  # 沿批次维拼接
            label_tensor = torch.cat([x for x in label_norm], dim=0)
            # 计算指标
            mse = torch.nn.functional.mse_loss(extract_tensor, label_tensor)
            psnr_value = psnr(extract_tensor, label_tensor, data_range=1.0)
            ssim_value = ssim(extract_tensor, label_tensor, data_range=1.0)
            iou_value = compute_iou(extract_tensor, label_tensor)
            dice_value = batch_soft_dice(label_tensor, extract_tensor)
            dice_mask_value = batch_mask_dice(label,extract_result)
            accuarcy_value = compute_pixel_accuracy_batch(extract_tensor, label_tensor)
            # LPIPS需要3通道
            extract_rgb = extract_tensor.repeat(1, 1, 1, 1)
            label_rgb = label_tensor.repeat(1, 1, 1, 1)
            
            # 将数据从 [0,1] 转换到 [-1,1]（LPIPS 要求）
            extract_tensor_lpips = extract_tensor * 2 - 1  # [0,1] -> [-1,1]
            label_tensor_lpips = label_tensor * 2 - 1      # [0,1] -> [-1,1]

            # 扩展为 3 通道（LPIPS 需要 RGB）
            extract_rgb = extract_tensor_lpips.repeat(1, 1, 1, 1)  # (N, 3, H, W)
            label_rgb = label_tensor_lpips.repeat(1, 1, 1, 1)      # (N, 3, H, W)

            # 计算 LPIPS
            lpips_value = loss_fn_lpips(extract_rgb, label_rgb).mean()
            
            torch.cuda.empty_cache()
            # 4. 将新指标添加到 loss_list
            loss_list.append([
                loss.item(), 
                mIOUm.item(), 
                mIOUum.item(),
                mse.item(),       # MSE (越小越好)
                psnr_value.item(), # PSNR (越大越好)
                ssim_value.item(), # SSIM (越大越好)
                lpips_value.item(), # LPIPS (越小越好
                iou_value.item(),
                dice_value.item(),
                dice_mask_value.item(),
                accuarcy_value.item(),
            ])
            if (i+1)%1==0 and (epoch+1)%1==0:
                # save data
                # extract_result_show = np.zeros(shape=(256, 256, 3), dtype=float) + target_data_o.squeeze().detach().to(
                #                                     'cpu').numpy().transpose(1, 2, 0)

                # label_result_show = np.zeros(shape=(256, 256, 3),dtype=float) + target_data_o.squeeze().detach().to(
                #                                         'cpu').numpy().transpose(1, 2, 0)

                black_canvas = np.zeros((256, 256, 3), dtype=np.float32)
                def shuffle_color(c, step=3):
                    shuf_c = []
                    for i in range(step):
                        num = 0
                        while i + num * step < len(c):
                            shuf_c.append(c[i + num * step])
                            num += 1
                    return shuf_c
                # 使用您原有的颜色生成逻辑
                r_colors = random_colors(len(extract_result))
                r_colors = shuffle_color(r_colors)
                
                # 在黑底上绘制笔画
                def draw_strokes_on_black(canvas, masks, colors):
                    """在黑底画布上绘制彩色笔画"""
                    result = canvas.copy()
                    ki = 0
                    for mask, color in zip(masks, colors):
                        if ki< srtrokes_num:
                            for c in range(3):
                                result[:,:,c] = np.where(mask <= 0.5, color[c], result[:,:,c])
                        ki+=1
                    return result
                
                # 生成黑底彩色笔画图
                black_with_strokes = draw_strokes_on_black(black_canvas, extract_result, r_colors)
                black_with_labels = draw_strokes_on_black(black_canvas, label, r_colors)
                
                # 反相得到白底彩色笔画图 (1 - 图像值)
                extract_result_show = 1.0 - black_with_strokes
                label_result_show = 1.0 - black_with_labels
                
                # 可选：叠加30%原始图像
                if False:  # 如需叠加原始图像，将此改为True
                    original_img = target_data_o.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                    extract_result_show = 0.3*original_img + 0.7*extract_result_show
                    label_result_show = 0.3*original_img + 0.7*label_result_show
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('stroke_label')
                title_list.append('stroke_extraction')

                save_list.append(reference_color.detach().to('cpu').repeat(2, 1, 1, 1))
                save_list.append(target_data_o.detach().to('cpu').repeat(2, 1, 1, 1))
                save_list.append(torch.from_numpy(label_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1))
                save_list.append(torch.from_numpy(extract_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(i)+str(epoch) +"-"+str(int(iou_value.item()*100.0))+'.bmp'),
                             nrow=int(save_list[0].size(0)))

        loss_value = np.mean(np.array(loss_list), axis=0)
        # 5. 打印时增加新指
        # loss_name = ['loss', 'mIOUm', 'mIOUum']
        loss_name = ['loss', 'mIOUm', 'mIOUum', 'MSE', 'PSNR', 'SSIM', 'LPIPS','IOU','Dice','Dicem','Accuracy']
        print(
            "[TEST][{}/{}], loss={:.7f}, mIOUm={:.7f}, mIOUum={:.7f}, "
            "MSE={:.7f}, PSNR={:.7f}, SSIM={:.7f}, LPIPS={:.7f},IOU={:.7f},Dice={:.7f},Dicem={:.7f}, Accuracy={:.7f}, time={:.7f}".format(
                i, len(test_loader), 
                loss_value[0], loss_value[1], loss_value[2],
                loss_value[3], loss_value[4], loss_value[5], loss_value[6],loss_value[7],loss_value[8],loss_value[9],loss_value[10],
                time.time() - start_time
            )
        )
        return loss_value, loss_name





if __name__ == '__main__':
    model = TrainExtractNet(save_path='out/03ExtractNet_ref_self', segNet_save_path='out/0SegNet_ref_self')
    #model.train_model(epochs=20, init_learning_rate=0.0001, batch_size=4, dataset=r'dataset_forSegNet_ExtractNet2_self')
    model.test_model(extract_save_path ='out/02ExtractNet_ref_self',dataset=r'dataset_forSegNet_ExtractNet0_self')