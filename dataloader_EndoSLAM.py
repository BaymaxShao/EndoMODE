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


class SeqData_ES(data.Dataset):
    def __init__(self, root, istrain=True, img_size=[320, 240], train_list='train_file.txt', test_list='test_file.txt', step=1):
        self.root = Path(root)
        self.istrain = istrain
        self.step = step
        objs_list = root + train_list if istrain else root + test_list
        self.objs = [self.root/folder.split('/')[1][:-1] for folder in open(objs_list)]
        self.get_samples()
        if self.istrain:
            self.resizer = transforms.Compose([transforms.Resize(img_size),
                                               transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                                      hue=0.1)])
        else:
            self.resizer = transforms.Resize(img_size)
        self.to_norm_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def get_samples(self):
        seq = []
        for obj in self.objs:
            frames = obj / 'Frames'
            opt_flows = obj/ 'OptFlow_4'
            imgs = sorted(frames.files('frame_*.jpg'))
            opts = sorted(opt_flows.files('*.jpg'))
            poses, poses_mat = self.get_poses(obj)
            for i in range(0, len(imgs)-self.step, self.step):
                sample = {'img1': imgs[i], 'img2': imgs[i+self.step], 'opt': opts[i//5],
                          'ref': [poses[i], poses_mat[i]], 'tar': [poses[i+self.step], poses_mat[i+self.step]]}
                seq.append(sample)

        self.samples = seq

    def get_poses(self, obj):
        poses = []
        locations = []
        rotations = []
        pose_name = Path('Traj.xlsx')
        pose_file = pd.read_excel(obj/pose_name, header=None)
        for pose in pose_file.values:
            poses.append(np.append(pose[6:10], pose[3:6]))
            locations.append(pose[3:6])
            rotations.append(pose[6:10])

        locations = np.array(locations[1:])  # in cm
        rotations = np.array(rotations[1:])
        poses = np.array(poses[1:])
        poses_mat = []
        for i in range(locations.shape[0]):
            r = R.from_quat(rotations[i]).as_matrix()
            T = np.concatenate((r, locations[i].reshape((3, 1))), 1)
            T = np.concatenate((T, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
            poses_mat.append(T)

        return poses, poses_mat

    def __getitem__(self, index):
        sample = self.samples[index]

        # Original Image Pair
        img1 = Image.open(sample['img1']).convert('RGB')
        img2 = Image.open(sample['img2']).convert('RGB')
        flow = Image.open(sample['opt']).convert('RGB')
        # print(sample['img1'], sample['img2'], sample['opt'])
        # print(sample['img1'], sample['img2'], sample['opt'])
        # left = 420  # 左上角x坐标
        # top = 0  # 左上角y坐标
        # right = 1500  # 右下角x坐标
        # bottom = 1080  # 右下角y坐标
        # img1 = img1.crop((left, top, right, bottom))
        img1 = self.resizer(img1)
        # img2 = img2.crop((left, top, right, bottom))
        img2 = self.resizer(img2)
        # flow = flow.crop((left, top, right, bottom))
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