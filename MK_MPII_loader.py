import scipy.io
import imageio
import os
import skimage
import skimage.io
import numpy as np
import random
from enum_JOINT import JOINT
from tqdm import tqdm
from vectormath import Vector2
from util import crop_image, generate_heatmap

class MPII:
    def __init__(self, batch_size, task='train'):
        self.mat = scipy.io.loadmat('data/mpii_human_pose_v1_u12_1.mat', squeeze_me = True, struct_as_record=False)
        self.image_num = len(getattr(self.mat['RELEASE'], 'img_train'))
        self.image_path = './data/images/'
        self.joint_num = len(JOINT)
        self.cursor = 0
        self.image_set = []
        self.batch_size = batch_size

        for img_idx in range(self.image_num):
            if task == 'train' and getattr(self.mat['RELEASE'], 'img_train')[img_idx] == 1:
                annorect = getattr(getattr(self.mat['RELEASE'], 'annolist'), 'annorect')
                if type(annorect) is np.ndarray:
                    for r_idx in range(len(annorect)):
                        self.image_set.append((img_idx, r_idx))
                else:
                    self.image_set.append(img_idx, -1)

    def __iter__(self):
        return self

    def __next__(self):
        batch_image = self.batch_image_set()
        return self.get_minibatch(batch_image)

    def batch_image_set(self):
        prev_cursor = self.cursor
        self.cursor += self.batch_size
        if self.cursor > len(self.image_set):
            self.image_set = random.shuffle(self.image_set)
            prev_cursor = 0
            self.cursor = self.batch_size
        return self.image_set[prev_cursor:self.cursor]

    def get_minibatch(self, batch_image):
        batch_rgb = np.ndarray(shape=(len(batch_image), 256, 256, 3), dtype=np.float32)
        batch_heatmap = np.zeros(shape=(len(batch_image), 64, 64, self.joint_num), dtype=np.float32)
        batch_keypoint = np.zeros(shape=(len(batch_image), self.joint_num, 2), dtype=np.float32)
        batch_activity = np.zeros(shape=(len(batch_image), 1), dtype=np.int32)

        for idx in range(len(batch_image)):
            batch_rgb[idx], batch_heatmap[idx], batch_keypoint[idx], batch_activity[idx] = self.get_data(batch_image[idx])

        return batch_rgb, batch_heatmap, batch_keypoint, batch_activity

    def get_data(self, idx):
        img_idx, r_idx = idx
        annolist = getattr(self.mat['RELEASE'], 'annolist')[img_idx]
        if r_idx != -1:
            annorect = getattr(annolist, 'annorect')[r_idx]
        else:
            annorect = getattr(annolist, 'annorect')

        image_name = self.image_path + getattr(getattr(annolist, 'image'), 'name')   
        image = skimage.img_as_float(skimage.io.imread(image_name))

        scale = annorect.scale
        hitbox = 200 * scale

        objpos = getattr(annorect, 'objpos')
        center = Vector2(getattr(objpos, 'x'), getattr(objpos, 'y'))
        ret_image = crop_image(image, center, 256)

        assert ret_image.shape == (256, 256, 3)

        ret_heatmap = np.zeros(shape=(64, 64, self.joint_num), dtype=np.float32)

        ret_keypoint = np.zeros(shape=(self.joint_num, 2))

        keypoints = annorect.annopoints.point

        for key_idx in range(keypoints.shape[0]):
            joint_id = keypoints[key_idx].id
            keypoint = Vector2(keypoints[key_idx].x, keypoints[key_idx].y)
            keypoint -= (center - hitbox / 2)

            if min(keypoint) < 0 or max(keypoint) >= 64:
                continue

            ret_heatmap[:, :, joint_id] = generate_heatmap(64, keypoint.y, keypoint.x)
            ret_keypoint[joint_id, :] = [keypoint.y, keypoint.x]

        act = getattr(self.mat['RELEASE'], 'act')[img_idx]
        ret_activity = act.act_id

        return ret_image, ret_heatmap, ret_keypoint, ret_activity

mpii = MPII(batch_size=100, task='train')