import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def data_augment(xb, yb):

    def rotate(xb, yb, angle):
        img_h, img_w = xb.shape[:-1]
        M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
        xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
        yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
        return xb, yb

    def blur(img):
        img = cv2.blur(img, (3, 3))
        return img

    def add_noise(img):
        for i in range(200):  # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 1
        return img

    if np.random.random() < 0.5:
        if np.random.random() < 0.25:
            xb, yb = rotate(xb, yb, 90)
        if np.random.random() < 0.25:
            xb, yb = rotate(xb, yb, 180)
        if np.random.random() < 0.25:
            xb, yb = rotate(xb, yb, 270)
        if np.random.random() < 0.25:
            xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
            yb = cv2.flip(yb, 1)
        if np.random.random() < 0.25:
            xb = blur(xb)
        if np.random.random() < 0.2:
            xb = add_noise(xb)
    return xb, yb

class data_loader():

    def __init__(self, datadir, batch_size=16, image_hw=(800, 600), n_class=1):
        self.train_files, self.test_files = [], []
        self.batch_size = batch_size
        self.n_class = n_class
        self.image_hw = image_hw

        # 读取训练图片文件路径
        train_dir = os.path.join(datadir, 'training')
        y_files = glob(train_dir+'/*_matte.png')
        for y_file in y_files:
            idx = (y_file.split(os.path.sep)[-1]).split('_')[0]
            x_file = os.path.join(train_dir, idx+'.png')
            self.train_files.append([x_file, y_file])

        # 读取测试图片路径
        test_dir = os.path.join(datadir, 'testing')
        y_files = glob(test_dir+'/*_matte.png')
        for y_file in y_files:
            idx = (y_file.split(os.path.sep)[-1]).split('_')[0]
            x_file = os.path.join(test_dir, idx+'.png')
            self.test_files.append([x_file, y_file])

    # 加载全部数据
    def get_data(self, set='train'):
        if set=='train':
            files = self.train_files
        else:
            files = self.test_files

        batch_x, batch_y = [], []
        for x_file, y_file in tqdm(files):
            # 读取图像
            x = cv2.imread(x_file) / 255.0
            y = cv2.imread(y_file, cv2.IMREAD_GRAYSCALE) / 255.0
            # 缩放
            if x.shape!=self.image_hw or y.shape!=self.image_hw:
                x = cv2.resize(x, self.image_hw[::-1])
                y = cv2.resize(y, self.image_hw[::-1])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.array(batch_x)
        batch_y = np.expand_dims(np.array(batch_y), -1)
        return batch_x, batch_y

    # 数据产生器
    def data_generator(self, set='train'):
        if set=='train':
            files = self.train_files
        else:
            files = self.test_files

        batch_x, batch_y = [], []
        while True:
            for x_file, y_file in files:
                # 读取图像
                x = cv2.imread(x_file) / 255.0
                y = cv2.imread(y_file, cv2.IMREAD_GRAYSCALE) / 255.0
                # 缩放
                if x.shape!=self.image_hw or y.shape!=self.image_hw:
                    x = cv2.resize(x, self.image_hw[::-1])
                    y = cv2.resize(y, self.image_hw[::-1])
                batch_x.append(x)
                batch_y.append(y)
                if len(batch_x)==self.batch_size:
                    batch_x = np.array(batch_x)
                    batch_y = np.expand_dims(np.array(batch_y), -1)
                    yield batch_x, batch_y
                    batch_x, batch_y = [], []