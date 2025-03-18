import json

import torch.utils.data as data
import numpy as np
from path import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2
from utils import *


def load_as_float(path):
    return Image.open(path)


class SeqData_SC(data.Dataset):
    def __init__(self, root, istrain=True, train_list='train_file.txt', test_list='test_file.txt', step=1):
        self.root = Path(root)
        self.istrain = istrain
        self.step = step
        objs_list = train_list if istrain else test_list
        self.objs = [self.root/folder[1:-1] for folder in open(objs_list)]
        self.get_samples()
        if self.istrain:
            self.resizer = transforms.Compose([transforms.Resize([320,256]), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
        else:
            self.resizer = transforms.Resize([320,256])
        self.to_norm_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def get_samples(self):
        seq = []
        for obj in self.objs:
            frames = obj / 'data/left'
            opt_flows = obj/ 'OptFlow_4'
            imgs = sorted(frames.files('*.png'))
            opts = sorted(opt_flows.files('*.jpg'))
            poses, poses_mat = self.get_poses(obj)
            for i in range(0, len(imgs)-self.step, self.step):
                sample = {'img1': imgs[i], 'img2': imgs[i+self.step], 'opt': opts[i//4],
                          'ref': [poses[i], poses_mat[i]], 'tar': [poses[i+self.step], poses_mat[i+self.step]]}
                seq.append(sample)

        self.samples = seq

    def get_poses(self, obj):
        poses = []
        locations = []
        rotations = []
        oks = []
        poses_mat = []
        frame_data = obj/'data/frame_data'
        pose_files = sorted(frame_data.files('frame_data*.json'))
        for pose_file in pose_files:
            T = json.load(open(pose_file))['camera-pose']
            T = np.array(T)
            poses_mat.append(T)
            rotation = R.from_matrix(T[:3,:3]).as_quat()
            rotations.append(rotation)
            location = T[:3,-1].reshape((1,3))
            locations.append(location)
            poses.append(list(rotation)+location.tolist()[0])

        locations = np.array(locations)  # in cm
        rotations = np.array(rotations)
        poses = np.array(poses)

        return poses, poses_mat

    def __getitem__(self, index):
        sample = self.samples[index]

        # Original Image Pair
        img1 = Image.open(sample['img1']).convert('RGB')
        img2 = Image.open(sample['img2']).convert('RGB')
        flow = Image.open(sample['opt']).convert('RGB')
        # print(sample['img1'], sample['img2'], sample['opt'])
        img1 = self.resizer(img1)
        img2 = self.resizer(img2)
        flow = self.resizer(flow)
        img1 = self.to_norm_tensor(np.array(img1))[:3, :, :]
        img2 = self.to_norm_tensor(np.array(img2))[:3, :, :]
        flow = self.to_norm_tensor(np.array(flow))[:3, :, :]

        # Absolute and Relative Pose Data
        ref_pos = sample['ref'][0]
        tar_pos = sample['tar'][0]
        ref_pos_mat = sample['ref'][1]
        tar_pos_mat = sample['tar'][1]
        ref_pos_mat = ref_pos_mat.astype(float)
        tar_pos_mat = tar_pos_mat.astype(float)
        rel_trans = np.matmul(np.linalg.inv(ref_pos_mat), tar_pos_mat)
        quat_diff = R.from_matrix(rel_trans[:3, :3]).as_quat()
        trans_diff = rel_trans[:3, -1]
        rel_pose = np.concatenate([quat_diff, trans_diff])
        ref_pos = ref_pos.astype(float)
        tar_pos = tar_pos.astype(float)

        return img1, img2, flow, ref_pos, tar_pos, rel_pose

    def __len__(self):
        return len(self.samples)