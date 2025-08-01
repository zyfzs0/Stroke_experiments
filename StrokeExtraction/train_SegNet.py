import torch
import torch.nn.functional as F
import os
import time
import torch.optim as optim
import torch.utils.data as data
from model.model_of_SegNet import SegNet
from load_data_for_SegNetExtractNet import SegNetExtractNetLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from utils import save_picture, random_colors, apply_stroke
from utils_loss_val import get_mean_IOU
import piq
from piq import psnr, ssim
import lpips
from CLDICE.cldice import soft_dice,soft_cldice,soft_dice_cldice

seg_colors = random_colors(33)
device = torch.device("cuda:0")
device0 = torch.device('cpu')
loss_fn_lpips = lpips.LPIPS(net='alex').to(device0)  # 使用 AlexNet 作为 LPIPS 的特征提取器


def min_max_normalize(x, target_data_o):
    """
    处理输入为 PyTorch Tensor 的情况
    """
    # 确保所有输入都是 NumPy 数组
    if isinstance(x, torch.Tensor):
        mask = x.detach().cpu().numpy().astype(np.float32)
    else:
        mask = x.astype(np.float32)
    
    if isinstance(target_data_o, torch.Tensor):
        target_np = target_data_o.detach().cpu().numpy()
    else:
        target_np = target_data_o
    
    # 创建白色背景
    white_bg = np.ones_like(target_np)
    
    # 应用掩码
    masked = target_np * (1 - mask) + white_bg * mask
    
    # 归一化处理
    c_min = masked.min()
    c_max = masked.max()
    normalized = (masked - c_min) / (c_max - c_min + 1e-8)
    
    return normalized.astype(np.float32)


def min_max_normalize_1(x, target_data_o):
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
    """计算 Dice 系数 (PyTorch 版本)"""
    smooth = 1e-5
    intersection = torch.sum(y_true * y_pred)  # 使用 torch.sum 而不是 np.sum
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
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
        yt = 1.0 - y_true[b].float()
        yp = 1.0 - y_pred[b].float()
        losses.append(soft_dice(yt, yp))
    return sum(losses)/B




def batch_mask_dice(y_true, y_pred):
    B = len(y_true)
    losses=[]
    # mask_losses = []
    for b in range(B):
        yt = 1.0 - y_true[b].float()
        yp = 1.0 - y_pred[b].float()
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


class TrainSegNet():
    '''
        train SegNet with the Train-Dataset
        validate SegNet with the Test-Dataset
    '''

    def __init__(self,  save_path=None):
        super().__init__()
        self.Out_path_train = os.path.join(save_path, 'train')
        self.Model_path = os.path.join(save_path, 'model')
        self.Out_path_val = os.path.join(save_path, 'val')
        self.Out_path_loss = os.path.join(save_path, 'loss')

        if not os.path.exists(self.Model_path):
            os.makedirs(self.Model_path)
        if not os.path.exists(self.Out_path_train):
            os.makedirs(self.Out_path_train)
        if not os.path.exists(self.Out_path_loss):
            os.makedirs(self.Out_path_loss)
        if not os.path.exists(self.Out_path_val):
            os.makedirs(self.Out_path_val)

        # SegNet
        self.seg_net = SegNet(out_feature=False)
        self.seg_net.to(device)

    def save_model_parameter(self, epoch):
        # save models
        state_stn = {'net': self.seg_net.state_dict(), 'start_epoch': epoch}
        torch.save(state_stn, os.path.join(self.Model_path, 'model.pth'))

    def train_model(self, epochs=40,  batch_size=16, init_learning_rate=0.001, dataset_path = None):
        self.batch_size = batch_size
        train_loader = data.DataLoader(SegNetExtractNetLoader(is_training=True, dataset_path=dataset_path), batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset_path), batch_size=batch_size)  # 24涓敤浜庢祴璇?

        optim_op = optim.Adam(self.seg_net.parameters(), lr=init_learning_rate, betas=(0.5, 0.999))
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
                self.__plot_loss(name+'.png', [train_data, test_data],
                               legend=['train', 'test'])
            # save models
            self.save_model_parameter(i)
            if (i+1)%2 == 0:
                lr_scheduler_op.step()
    
    def test_model(self, segnet_save_path =None,dataset=None,batch_size=1):
        self.batch_size = batch_size
        # load parameters of SegNet
        seg_model_path = os.path.join(segnet_save_path, 'model', 'model.pth')
        state = torch.load(seg_model_path)
        self.seg_net.load_state_dict(state['net'])
        self.seg_net.to(device).eval().requires_grad_(False)
        # dataset
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset), batch_size=batch_size)  # 24涓敤浜庢祴璇?
        test_history_loss = []
        test_loss, loss_name = self.__val_epoch_1(20, test_loader)
        test_history_loss.append(test_loss)
        for index, name in enumerate(loss_name):
            test_data = [x[index] for x in test_history_loss]
            self.__plot_loss(name+'.png', [test_data],
                            legend=['test'])
    
    def __plot_loss(self, name, loss, legend, save=True):
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
            save_path = os.path.join(self.Out_path_loss, name)
            plt.savefig(save_path)
        else:
            plt.show()

    def __train_epoch(self, epoch, train_loader, optim_opWhole):
        epoch += 1
        self.seg_net.train()
        loss_list = []
        start_time = time.time()

        for i, batch_sample in enumerate(train_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            reference_color = batch_sample['reference_color'].float().to(device)
            label_seg = batch_sample['label_seg'].float().to(device)
            target_data = batch_sample['target_data'].float().to(device)

            seg_out = self.seg_net(target_data, reference_color)

            seg_loss = F.binary_cross_entropy(F.sigmoid(seg_out), label_seg)
            seg_result_ = (F.sigmoid(seg_out).detach() > 0.5)
            mean_iou = get_mean_IOU(seg_result_, label_seg)
            optim_opWhole.zero_grad()
            seg_loss.backward()
            optim_opWhole.step()
            torch.cuda.empty_cache()
            loss_list.append([seg_loss.item(), mean_iou.item()])

            if i%50==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('label_seg')
                title_list.append('seg_result')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))

                save_list.append(self.__to_color(label_seg))
                save_list.append(self.__to_color(seg_result_))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_train, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))


        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['seg_loss', 'mean_iou']
        print(
            "[TRAIN][{}/{}],   seg_loss={:.7f}, mean_iou={:.7f}, time={:.7f}".format(i, len(train_loader), loss_value[0], loss_value[1], time.time() - start_time))

        return loss_value, loss_name
    
    def __val_epoch(self, epoch, test_loader):
        epoch += 1
        self.seg_net.eval()
        loss_list = []

        start_time = time.time()
        for i, batch_sample in enumerate(test_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            reference_color = batch_sample['reference_color'].float().to(device)
            label_seg = batch_sample['label_seg'].float().to(device)
            target_data = batch_sample['target_data'].float().to(device)
            target_stroke_data = batch_sample['target_single_stroke'].float().to(device)
            seg_out = self.seg_net(target_data, reference_color)
            srtrokes_num =batch_sample['stroke_num'].to(device0)
            seg_loss = F.binary_cross_entropy(F.sigmoid(seg_out), label_seg)
            seg_result_ = (F.sigmoid(seg_out).detach() > 0.5)
            mean_iou = get_mean_IOU(seg_result_, label_seg)
            target_data_o = target_data.clone()
            # print(srtrokes_num)
            torch.cuda.empty_cache()
            loss_list.append([seg_loss.item(), mean_iou.item()])
            

            if (i+1)%1==0 and epoch%1==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('label_seg')
                title_list.append('seg_result')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))

                save_list.append(self.__to_color(label_seg))
                save_list.append(self.__to_color(seg_result_))
                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))
        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['seg_loss', 'mean_iou']
        print(
            "[TEST][{}/{}],   seg_loss={:.7f}, mean_iou={:.7f}, time={:.7f}".format(i, len(test_loader),
                                                                                     loss_value[0], loss_value[1],
                                                                                     time.time() - start_time))

        return loss_value, loss_name
    
    
    def __val_epoch_1(self, epoch, test_loader):
        epoch += 1
        self.seg_net.eval()
        loss_list = []

        start_time = time.time()
        for i, batch_sample in enumerate(test_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            reference_color = batch_sample['reference_color'].float().to(device)
            label_seg = batch_sample['label_seg'].float().to(device)
            target_data = batch_sample['target_data'].float().to(device)
            target_stroke_data = batch_sample['target_single_stroke'].float().to(device)
            seg_out = self.seg_net(target_data, reference_color)
            srtrokes_num =batch_sample['stroke_num'].to(device0)
            seg_loss = F.binary_cross_entropy(F.sigmoid(seg_out), label_seg)
            seg_result_ = (F.sigmoid(seg_out).detach() > 0.5)
            mean_iou = get_mean_IOU(seg_result_, label_seg)
            target_data_o = target_data.clone()
            #print(target_data_o.shape,seg_result_.shape)
            target_data_o = target_data_o.squeeze(0)
            extract_result = seg_result_.squeeze(0)[:srtrokes_num[0]]
            label = label_seg.squeeze(0)[:srtrokes_num[0]]
            # print(label.shape)
            extract_result = list(extract_result.unbind(dim=0))  # 沿 dim=0 拆分，转为列表
            label = list(label.unbind(dim=0))            # 同上
            # print(len(extract))
            extract_result_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in extract_result]
            label_norm = [torch.from_numpy(min_max_normalize(x,target_data_o)).float() for x in label]
            #print(len(extract_result_norm),extract_result_norm[0].shape)
            # 转换为PIQ需要的格式 (N, 1, H, W)
            extract_tensor = torch.stack([x for x in extract_result_norm], dim=0)  # 沿批次维拼接
            label_tensor = torch.stack([x for x in label_norm], dim=0)
            # 计算指标
            #print(extract_tensor.shape)
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
            # print(srtrokes_num)
            torch.cuda.empty_cache()
            loss_list.append([seg_loss.item(), mean_iou.item(),mse.item(),       # MSE (越小越好)
                    psnr_value.item(), # PSNR (越大越好)
                    ssim_value.item(), # SSIM (越大越好)
                    lpips_value.item(),# LPIPS (越小越好)
                    iou_value.item(),
                    dice_value.item(),
                    dice_mask_value.item(),
                    accuarcy_value.item(),])
            

            if (i+1)%1==0 and epoch%1==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('label_seg')
                title_list.append('seg_result')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))

                save_list.append(self.__to_color_1(label_seg,srtrokes_num))
                save_list.append(self.__to_color_1(seg_result_,srtrokes_num))
                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))
        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['seg_loss', 'mean_iou', 'MSE', 'PSNR', 'SSIM', 'LPIPS','IOU','Dice','Dicem','Accuracy']
        print(
            "[TEST][{}/{}],   seg_loss={:.7f}, mean_iou={:.7f},MSE={:.7f}, PSNR={:.7f}, SSIM={:.7f}, LPIPS={:.7f},IOU={:.7f},Dice={:.7f},Dicem={:.7f}, Accuracy={:.7f} time={:.7f}".format(i, len(test_loader),
                                                                                     loss_value[0], loss_value[1],loss_value[2],loss_value[3], loss_value[4], loss_value[5], loss_value[6],loss_value[7],loss_value[8],loss_value[9],
                                                                                     time.time() - start_time))

        return loss_value, loss_name

    def __to_color_1(self, seg_result,srtrokes_num):
        '''
        Coloring the results of SegNet
        '''
        images = []
        for i in range(self.batch_size):
            image = np.zeros(shape=(256, 256, 3) ,dtype=float)
            for j in range(srtrokes_num[0]):
                image = apply_stroke(image, seg_result[i, j].detach().to('cpu').numpy()<=0.5, seg_colors[j])
            images.append(1.0 - image.transpose((2,0,1)))
        return torch.from_numpy(np.array(images))
    
    def __to_color(self, seg_result):
        '''
        Coloring the results of SegNet
        '''
        images = []
        for i in range(self.batch_size):
            image = np.zeros(shape=(256, 256, 3) ,dtype=float)
            for j in range(7):
                image = apply_stroke(image, seg_result[i, j].detach().to('cpu').numpy()<=0.5, seg_colors[j])
            images.append(image.transpose((2,0,1)))
        return torch.from_numpy(np.array(images))


if __name__ == '__main__':
    model = TrainSegNet(save_path=os.path.join('out0/7_SegNet_blank_self'))
    # model.train_model(epochs=1, init_learning_rate=0.0001, batch_size=4, dataset_path=r'dataset_forSegNet_ExtractNet7_blank_self')
    model.test_model(segnet_save_path ='out0/7_SegNet_vblank_self',dataset=r'dataset_forSegNet_ExtractNet7_blank_self',batch_size=1)
