import torch.utils.data as data
import numpy as np
from path import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2

def load_as_float(path):
    return Image.open(path)

class SeqData(data.Dataset):
    def __init__(self, root, istrain=True, img_size=224, train_list='train_file.txt', test_list='test_file.txt', step=1):
        self.root = Path(root)
        self.istrain = istrain
        self.step = step
        objs_list = train_list if istrain else test_list
        self.objs = [self.root/folder.split('/')[0] for folder in open(objs_list)]
        if self.istrain:
            self.resizer = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
        else:
            self.resizer = transforms.Resize((img_size, img_size))
        self.to_norm_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def get_samples(self, objs):
        seq = []
        for obj in objs:
            imgs = sorted(objs.files('*.jpg'))
            poses, poses_mat = self.get_poses(obj)
            for i in range(0, len(imgs)-self.step, self.step):
                sample = {'img1': imgs[i], 'img2': imgs[i+self.step],
                          'ref': [poses[i], poses_mat[i]], 'tar': [poses[i+self.step], poses_mat[i+self.step]]}
                seq.append(sample)
        self.samples = seq

    def get_poses(self, obj):
        poses = []
        locations = []
        rotations = []
        pose_name = Path('pose.xlsx')
        pose_file = pd.read_excel(obj/pose_name)
        for pose in pose_file.values:
            poses.append(pose[5:12])
            locations.append(pose[9:12])
            rotations.append(pose[5:9])
        locations = np.array(locations)
        rotations = np.array(rotations)

        # r = R.from_quat(rotations).as_dcm()
        r = R.from_quat(rotations).as_matrix()

        TM = np.eye(4)
        TM[1, 1] = -1

        poses_mat = []
        for i in range(locations.shape[0]):
            ri = r[i]  # np.linalg.inv(r[0])
            Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
            Pi_left = TM @ Pi @ TM
            poses_mat.append(Pi_left)

        return poses, np.array(poses_mat)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Optical Flow Estimation
        img1_cv = cv2.imread(str(sample['img1']))
        img2_cv = cv2.imread(str(sample['img2']))
        mask = np.zeros_like(img1_cv)
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow_xy = TVL1.calc(img1_gray, img2_gray, None)

        mag, ang = cv2.cartToPolar(flow_xy[:, :, 0], flow_xy[:, :, 1])
        mask[:, :, 0] = ang * 180 / np.pi / 2  # 角度
        mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Original Image Pair
        img1 = self.resizer(Image.open(sample['img1']).convert('RGB'))
        img2 = self.resizer(Image.open(sample['img2']).convert('RGB'))
        img1 = self.to_norm_tensor(np.array(img1))[:3, :, :]
        img2 = self.to_norm_tensor(np.array(img2))[:3, :, :]

        # Absolute and Relative Pose Data
        ref_pos = sample['ref'][0]
        tar_pos = sample['tar'][0]
        ref_pos_mat = sample['ref'][1]
        tar_pos_mat = sample['tar'][1]
        rel_trans = np.matmul(np.linalg.inv(ref_pos_mat), tar_pos_mat)
        quat_diff = R.from_matrix(rel_trans[:3, :3]).as_quat()
        trans_from_quat = rel_trans[:3, -1]
        rel_pose = np.concatenate([trans_from_quat, quat_diff])

        return img1, img2, ref_pos, tar_pos, ref_pos_mat, tar_pos_mat, rel_pose



