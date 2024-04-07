'''
Here are the functions for:
- Calculation of metrics
- Transformation from log(q) to quaternion
- Transformation from quaternion to homogeneous matrix
'''


import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def logq_to_quaternion(q):
    # return: quaternion with w, x, y, z
    # from geomap paper
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((q, torch.cos(n)), dim=1)
    return q


def quat2T(q):
    rotations = q[0, :4]
    r = R.from_quat(rotations).as_matrix()
    trans = q[0, 4:]
    T = np.concatenate((r, trans.reshape((3, 1))), 1)
    T = np.concatenate((T, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
    return T


def ate(pose_gt, pose_pred):
    pos_gt = pose_gt[:3, -1]
    pos_pred = pose_pred[:3, -1]
    print('position_gt:{}'.format(str(pos_gt)))
    print('position_pred:{}'.format(str(pos_pred)))
    return np.linalg.norm(np.array(pos_gt) - np.array(pos_pred))


def ce(pose_gt, pose_pred):
    rot_gt = np.array(pose_gt[:3, :3])
    rot_pred = np.array(pose_pred[:3, :3])
    euler_gt = R.from_matrix(rot_gt).as_euler('xyz', degrees=True)
    euler_pred = R.from_matrix(rot_pred).as_euler('xyz', degrees=True)
    print('rotation_gt:{}'.format(str(euler_gt)))
    print('rotation_pred:{}'.format(str(euler_pred)))
    ce = ((1-np.cos(euler_pred[0]-euler_gt[0]))+(1-np.cos(euler_pred[1]-euler_gt[1]))+(1-np.cos(euler_pred[2]-euler_gt[2]))) / 3
    return ce, euler_pred, euler_gt


def rte(rel_gt, rel_pred):
    rel = np.matmul(np.linalg.inv(rel_gt), rel_pred)
    trans_rel = rel[:3, -1]
    return np.linalg.norm(trans_rel)


def rot(rel_gt, rel_pred):
    rot_rel = np.matmul(np.linalg.inv(rel_gt)[:3, :3], rel_pred[:3, :3])
    trace_rel = np.trace(rot_rel)
    dis = abs(np.arccos((trace_rel - 1)/2))
    return dis*180/np.pi