import os
from glob import glob
import threading
import multiprocessing
import signal
import sys
from datetime import datetime
import platform

import tensorflow as tf
import numpy as np
import cairosvg
from PIL import Image
import io
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch
import cv2

from ops import *


class BatchManager(object):
    def __init__(self, config):
        self.root = config.data_path
        self.rng = np.random.RandomState(config.random_seed)

        self.paths = sorted(glob("{}/train/*.{}".format(self.root, 'svg_pre')))
        self.test_paths = sorted(glob("{}/test/*.{}".format(self.root, 'svg_pre')))
        assert(len(self.paths) > 0 and len(self.test_paths) > 0)
        
        self.batch_size = config.batch_size
        self.height = config.height
        self.width = config.width

        self.is_pathnet = (config.archi == 'path')
        if self.is_pathnet:
            feature_dim = [self.height, self.width, 2]
            label_dim = [self.height, self.width, 1]
        else:
            feature_dim = [self.height, self.width, 1]
            label_dim = [self.height, self.width, 1]

        self.capacity = 10000
        self.q = tf.FIFOQueue(self.capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = config.num_worker
        # np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, w, h, is_pathnet):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    if is_pathnet:
                        x_, y_ = preprocess_path(paths[id], w, h, rng)
                    else:
                        x_, y_ = preprocess_overlap(paths[id], w, h, rng)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self.sess, 
                                                self.enqueue,
                                                self.coord,
                                                self.paths,
                                                self.rng,
                                                self.x,
                                                self.y,
                                                self.width,
                                                self.height,
                                                self.is_pathnet)
                                          ) for i in range(self.num_threads)]

        # define signal handler
        def signal_handler(signum, frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False
        qs = 0
        while qs < (self.capacity*0.8):
            qs = self.sess.run(self.q.size())
        print('%s: q size %d' % (datetime.now(), qs))

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def test_batch(self):
        x_list, y_list = [], []
        for i, file_path in enumerate(self.test_paths):
            if self.is_pathnet:
                x_, y_ = preprocess_path(file_path, self.width, self.height, self.rng)
            else:
                x_, y_ = preprocess_overlap(file_path, self.width, self.height, self.rng)
            x_list.append(x_)
            y_list.append(y_)
            if i % self.batch_size == self.batch_size-1:
                yield np.array(x_list), np.array(y_list)
                x_list, y_list = [], []

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    def sample(self, num):
        idx = self.rng.choice(len(self.paths), num).tolist()
        return [self.paths[i] for i in idx]

    def random_list(self, num):
        x_list = []
        xs, ys = [], []
        file_list = self.sample(num)
        for file_path in file_list:
            if self.is_pathnet:
                x, y = preprocess_path(file_path, self.width, self.height, self.rng)
            else:
                x, y = preprocess_overlap(file_path, self.width, self.height, self.rng)
            x_list.append(x)

            if self.is_pathnet:
                b_ch = np.zeros([self.height,self.width,1])
                xs.append(np.concatenate((x*255, b_ch), axis=-1))
            else:
                xs.append(x*255)
            ys.append(y*255)
            
        return np.array(x_list), np.array(xs), np.array(ys), file_list

    def read_svg(self, file_path):
        with open(file_path, 'r') as f:
            svg = f.read()

        r = 0
        s = [1, -1]
        t = [0, -900]
        # if transform:
        #     r = rng.randint(-45, 45)
        #     # s_sign = rng.choice([1, -1], 1)[0]
        #     s_sign = -1
        #     s = 1.75 * rng.random_sample(2) + 0.25 # [0.25, 2)
        #     s[1] = s[1] * s_sign
        #     t = rng.randint(-100, 100, 2)
        #     if s_sign == 1:
        #         t[1] = t[1] + 124
        #     else:
        #         t[1] = t[1] - 900

        svg = svg.format(w=self.width, h=self.height, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity

        path_list = []        
        svg_xml = et.fromstring(svg)

        sys_name = platform.system()
        if sys_name == 'Windows':
            num_paths = len(svg_xml[0]._children)
        else:
            num_paths = len(svg_xml[0])

        for i in range(num_paths):
            svg_xml = et.fromstring(svg)

            if sys_name == 'Windows':
                svg_xml[0]._children = [svg_xml[0]._children[i]]
            else:
                svg_xml[0][0] = svg_xml[0][i]
                del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = (np.array(y_img)[:,:,3] > 0)            
            path_list.append(path)

        return s, num_paths, path_list




class BatchManager_new(object):
    def __init__(self, config):
        self.config = config
        self.root = config.data_path
        self.rng = np.random.RandomState(config.random_seed)
        
        # 加载CSV文件
        if config.is_training:
            csv_path = '/remote-home/zhangxinyue/stroke_segmentation/split_data_by_character/train_metadata.csv'
        else:
            csv_path = '/remote-home/zhangxinyue/stroke_segmentation/split_data_by_character/test_metadata.csv'
        
        self.df = pd.read_csv(csv_path)
        self.character_dir = '/remote-home/zhangxinyue/stroke_segmentation/pixel_all_characters'
        self.stroke_dir = '/remote-home/zhangxinyue/stroke_segmentation/pixel_all_strokes'
        
        # 初始化数据分组
        self.grouped_df = self.df.groupby('character')
        self.valid_characters = []
        self._validate_data_pairs()
        
        # 数据参数
        self.batch_size = config.batch_size
        self.height = config.height
        self.width = config.width
        self.is_pathnet = (config.archi == 'path')
        
        # 定义数据维度
        if self.is_pathnet:
            feature_dim = [self.height, self.width, 2]
            label_dim = [self.height, self.width, 1]
        else:
            feature_dim = [self.height, self.width, 1]
            label_dim = [self.height, self.width, 1]
            
        # 初始化TensorFlow队列
        self.capacity = 10000
        self.q = tf.FIFOQueue(self.capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = config.num_worker
        
        # 数据转换
        self.character_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)
        ])
        self.stroke_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)
        ])

    def _validate_data_pairs(self):
        """验证字符和笔画图像是否匹配"""
        for char_id, group in self.grouped_df:
            valid = True
            # 检查字符图像是否存在
            char_path = os.path.join(self.character_dir, f"{char_id}.jpg")
            if not os.path.exists(char_path):
                print(f"Warning: Missing character image {char_path}")
                valid = False
                continue
                
            # 检查所有笔画图像是否存在
            for _, row in group.iterrows():
                stroke_path = os.path.join(self.stroke_dir, f"{row['target']}.jpg")
                if not os.path.exists(stroke_path):
                    print(f"Warning: Missing stroke image {stroke_path}")
                    valid = False
                    break
            
            if valid:
                self.valid_characters.append(char_id)
        
        assert len(self.valid_characters) > 0, "No valid character-stroke pairs found!"

    def _load_images(self, char_path, stroke_path):
        """加载字符和笔画图像"""
        char_img = Image.open(char_path).convert('L')
        stroke_img = Image.open(stroke_path).convert('L')
        
        # 应用转换
        char_img = self.character_transform(char_img).numpy()[0]  # (H, W)
        stroke_img = self.stroke_transform(stroke_img).numpy()[0]  # (H, W)
        
        return char_img, stroke_img

    def preprocess_data(self, char_path, stroke_path):
        """预处理数据"""
        char_img, stroke_img = self._load_images(char_path, stroke_path)
        
        # 归一化
        char_img = char_img / 255.0
        stroke_img = stroke_img / 255.0

        if self.is_pathnet:
            # pathnet模式：输入是2通道（字符+笔画）
            x = np.zeros((self.height, self.width, 2), dtype=np.float32)
            x[:, :, 0] = char_img
            x[:, :, 1] = stroke_img
            y = np.expand_dims(stroke_img, axis=-1)
        else:
            # overlap模式：输入是单通道字符
            x = np.expand_dims(char_img, axis=-1)
            y = np.expand_dims(stroke_img, axis=-1)
        
        return x, y

    def start_thread(self, sess):
        """启动数据加载线程"""
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))
        
        self.sess = sess
        self.coord = tf.train.Coordinator()

        def load_n_enqueue(sess, enqueue, coord, grouped_df, valid_characters, rng,
                         x, y, w, h, is_pathnet, character_dir, stroke_dir):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    # 随机选择一个字符
                    char_id = rng.choice(valid_characters)
                    group = grouped_df.get_group(char_id)
                    
                    # 加载字符图像
                    char_path = os.path.join(character_dir, f"{char_id}.jpg")
                    
                    # 随机选择一个笔画
                    stroke_row = group.sample(n=1).iloc[0]
                    stroke_path = os.path.join(stroke_dir, f"{stroke_row['target']}.jpg")
                    
                    # 预处理并加入队列
                    x_, y_ = self.preprocess_data(char_path, stroke_path)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        self.threads = [threading.Thread(
            target=load_n_enqueue,
            args=(self.sess, self.enqueue, self.coord,
                  self.grouped_df, self.valid_characters, self.rng,
                  self.x, self.y, self.width, self.height, self.is_pathnet,
                  self.character_dir, self.stroke_dir)
        ) for _ in range(self.num_threads)]

        # 信号处理
        def signal_handler(signum, frame):
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        for t in self.threads:
            t.start()

        # 等待队列填充
        g = tf.get_default_graph()
        g._finalized = False
        qs = 0
        while qs < (self.capacity*0.8):
            qs = self.sess.run(self.q.size())
        print('%s: q size %d' % (datetime.now(), qs))

    def stop_thread(self):
        """停止数据加载线程"""
        g = tf.get_default_graph()
        g._finalized = False

        if hasattr(self, 'coord'):
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)

    def test_batch(self):
        """生成测试数据"""
        x_list, y_list = [], []
        for char_id in self.valid_characters[:100]:  # 测试前100个字符
            group = self.grouped_df.get_group(char_id)
            char_path = os.path.join(self.character_dir, f"{char_id}.jpg")
            
            for _, row in group.iterrows():
                stroke_path = os.path.join(self.stroke_dir, f"{row['target']}.jpg")
                x_, y_ = self.preprocess_data(char_path, stroke_path)
                x_list.append(x_)
                y_list.append(y_)
                
                if len(x_list) == self.batch_size:
                    yield np.array(x_list), np.array(y_list)
                    x_list, y_list = []

    def batch(self):
        """获取一个batch的数据"""
        return self.q.dequeue_many(self.batch_size)

    def sample(self, num):
        """随机采样num个字符"""
        char_ids = self.rng.choice(self.valid_characters, num).tolist()
        samples = []
        for char_id in char_ids:
            group = self.grouped_df.get_group(char_id)
            stroke_row = group.sample(n=1).iloc[0]
            samples.append((
                os.path.join(self.character_dir, f"{char_id}.jpg"),
                os.path.join(self.stroke_dir, f"{stroke_row['target']}.jpg")
            ))
        return samples

    def random_list(self, num):
        """随机采样并返回处理后的数据"""
        x_list = []
        xs, ys = [], []
        file_pairs = self.sample(num)
        
        for char_path, stroke_path in file_pairs:
            x, y = self.preprocess_data(char_path, stroke_path)
            x_list.append(x)

            if self.is_pathnet:
                b_ch = np.zeros([self.height, self.width, 1])
                xs.append(np.concatenate((x*255, b_ch), axis=-1))
            else:
                xs.append(x*255)
            ys.append(y*255)
            
        return np.array(x_list), np.array(xs), np.array(ys), file_pairs

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass


def preprocess_path(file_path, w, h, rng):
    with open(file_path, 'r') as f:
        svg = f.read()

    r = 0
    s = [1, -1]
    t = [0, -900]
    # if transform:
    #     r = rng.randint(-45, 45)
    #     # s_sign = rng.choice([1, -1], 1)[0]
    #     s_sign = -1
    #     s = 1.75 * rng.random_sample(2) + 0.25 # [0.25, 2)
    #     s[1] = s[1] * s_sign
    #     t = rng.randint(-100, 100, 2)
    #     if s_sign == 1:
    #         t[1] = t[1] + 124
    #     else:
    #         t[1] = t[1] - 900

    svg = svg.format(w=w, h=h, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity

    # while True:
    svg_xml = et.fromstring(svg)

    sys_name = platform.system()
    if sys_name == 'Windows':
        path_id = rng.randint(len(svg_xml[0]._children))
        svg_xml[0]._children = [svg_xml[0]._children[path_id]]
    else:
        path_id = rng.randint(len(svg_xml[0]))
        svg_xml[0][0] = svg_xml[0][path_id]
        del svg_xml[0][1:]
    svg_one = et.tostring(svg_xml, method='xml')

    # leave only one path
    y_png = cairosvg.svg2png(bytestring=svg_one)
    y_img = Image.open(io.BytesIO(y_png))
    y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity # [0,1]

    pixel_ids = np.nonzero(y)
    # if len(pixel_ids[0]) == 0:
    #     continue
    # else:
    #     break            

    # select arbitrary marking pixel
    point_id = rng.randint(len(pixel_ids[0]))
    px, py = pixel_ids[0][point_id], pixel_ids[1][point_id]

    y = np.reshape(y, [h, w, 1])
    x = np.zeros([h, w, 2])
    x[:,:,0] = s
    x[px,py,1] = 1.0

    # # debug
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.subplot(223)
    # plt.imshow(np.concatenate((x, np.zeros([h, w, 1])), axis=-1))
    # plt.subplot(224)
    # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
    # plt.show()

    return x, y

def preprocess_overlap(file_path, w, h, rng):
    with open(file_path, 'r') as f:
        svg = f.read()
    
    r = 0
    s = [1, -1]
    t = [0, -900]
    # if transform:
    #     r = rng.randint(-45, 45)
    #     # s_sign = rng.choice([1, -1], 1)[0]
    #     s_sign = -1
    #     s = 1.75 * rng.random_sample(2) + 0.25 # [0.25, 2)
    #     s[1] = s[1] * s_sign
    #     t = rng.randint(-100, 100, 2)
    #     if s_sign == 1:
    #         t[1] = t[1] + 124
    #     else:
    #         t[1] = t[1] - 900

    svg = svg.format(w=w, h=h, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity

    # while True:
    path_list = []        
    svg_xml = et.fromstring(svg)

    sys_name = platform.system()
    if sys_name == 'Windows':
        num_paths = len(svg_xml[0]._children)
    else:
        num_paths = len(svg_xml[0])

    for i in range(num_paths):
        svg_xml = et.fromstring(svg)

        if sys_name == 'Windows':
            svg_xml[0]._children = [svg_xml[0]._children[i]]
        else:
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]
        svg_one = et.tostring(svg_xml, method='xml')

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one)
        y_img = Image.open(io.BytesIO(y_png))
        path = (np.array(y_img)[:,:,3] > 0)            
        path_list.append(path)

    y = np.zeros([h, w], dtype=np.int)
    for i in range(num_paths-1):
        for j in range(i+1, num_paths):
            intersect = np.logical_and(path_list[i], path_list[j])
            y = np.logical_or(intersect, y)

    x = np.expand_dims(s, axis=-1)
    y = np.expand_dims(y, axis=-1)

    # # debug
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(img)
    # plt.subplot(132)
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.subplot(133)
    # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
    # plt.show()

    return x, y

def main(config):
    prepare_dirs_and_logger(config)
    batch_manager = BatchManager(config)
    preprocess_path('data/ch/train/11904.svg_pre', 64, 64, batch_manager.rng)
    preprocess_overlap('data/ch/train/11904.svg_pre', 64, 64, batch_manager.rng)

    # thread test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    batch_manager.start_thread(sess)

    x, y = batch_manager.batch()
    if config.data_format == 'NCHW':
        x = nhwc_to_nchw(x)
    x_, y_ = sess.run([x, y])
    batch_manager.stop_thread()

    if config.data_format == 'NCHW':
        x_ = x_.transpose([0, 2, 3, 1])

    if config.archi == 'path':
        b_ch = np.zeros([config.batch_size,config.height,config.width,1])
        x_ = np.concatenate((x_*255, b_ch), axis=-1)
    else:
        x_ = x_*255
    y_ = y_*255

    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))
    save_image(y_, '{}/y_fixed.png'.format(config.model_dir))


    # random pick from parameter space
    x_samples, x_gt, y_gt, sample_list = batch_manager.random_list(8)
    save_image(x_gt, '{}/x_gt.png'.format(config.model_dir))
    save_image(y_gt, '{}/y_gt.png'.format(config.model_dir))

    with open('{}/sample_list.txt'.format(config.model_dir), 'w') as f:
        for sample in sample_list:
            f.write(sample+'\n')

    print('batch manager test done')

if __name__ == "__main__":
    from config import get_config
    from utils import prepare_dirs_and_logger, save_config, save_image

    config, unparsed = get_config()
    setattr(config, 'archi', 'path') # overlap
    setattr(config, 'dataset', 'ch')
    setattr(config, 'width', 64)
    setattr(config, 'height', 64)

    main(config)