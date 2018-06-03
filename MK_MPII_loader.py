import scipy.io
import imageio
import os
import skimage
import skimage.io
import numpy as np
from enum_JOINT import JOINT
from tqdm import tqdm
from util import crop_image

class MPII:
    def __init__(self, batch_size, task='train'):
        self.mat = scipy.io.loadmat('data/mpii_human_pose_v1_u12_1.mat', squeeze_me = True, struct_as_record=False)
        self.total_num = len(getattr(self.mat['RELEASE'], 'img_train'))
        self.image_path = './data/images/'
        self.joint_num = len(JOINT)
        self.cursor = 0
        self.image_set = []
        self.batch_size = batch_size

        for img_idx in range(self.total_num):
            if task == 'train' and getattr(self.mat['RELEASE'], 'img_train') == 1:
                self.image_set.append(img_idx)
            if task == 'test' and getattr(self.mat['RELEASE'], 'img_train') == 0:
                self.image_set.append(img_idx)

    def __iter__(self):
        return self

    def __next__(self):
        batch_image = self.batch_image_set()
        return self.get_minibatch(batch_image)

    def batch_image_set(self):
        prev_cursor = self.cursor
        self.cursor += self.batch_size
        return self.image_set[prev_cursor:self.cursor]

    def get_minibatch(self, batch_image):
        batch_rgb = np.ndarray(shape=(len(batch_image), 256, 256, 3), dtype=np.float32)
        batch_heatmap = np.zeros(shape=(len(batch_image), 64, 64, self.joint_num), dtype=np.float32)
        batch_keypoint = np.zeros(shape=(len(batch_image), self.joint_num, 2), dtype=np.float32)
        batch_activity = np.zeros(shape=(len(batch_image), 1), dtype=np.int32)

        for idx in range(len(batch_image)):
            batch_rgb[idx], batch_heatmap[idx], batch_keypoint[idx], batch_activity[idx] = self.get_data(batch_image[idx])

        return batch_rgb, batch_heatmap, batch_keypoint, batch_activity

    def get_data(self, img_idx):
        annolist = getattr(self.mat['RELEASE'], 'annolist')[img_idx]
        image_name = self.image_path + getattr(getattr(annolist, 'image'), 'name')   
        act = getattr(self.mat['RELEASE'], 'act')[img_idx]
        
        image = skimage.img_as_float(skimage.io.imread(image_name))
        objpos = getattr(getattr(annolist, 'annorect')[0], 'objpos') #r_idx = 0인 사람의 objpos 뽑음
        posX, posY = getattr(objpos, 'x'), getattr(objpos, 'y')
        image = crop_image(image, (posX, posY), 256)

        heatmap = np.zeros(shape=(64, 64, self.joint_num), dtype=np.float32)

        keypoint = np.zeros(shape=(self.joint_num, 2))
        

    

mpii = MPII(batch_size=100, task='train')