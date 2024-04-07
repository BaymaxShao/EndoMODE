'''
This is the code for the visualization of our results.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def cal_distance(gt, pred):
    return math.sqrt((float(gt[0]) - float(pred[0])) ** 2 + (
                float(gt[0]) - float(pred[0])) ** 2 + (
                          float(gt[0]) - float(pred[0])) ** 2)


def caldir(yaw, pitch, roll):
    # Define the unit of direction vector
    direction_vector = np.array([1, 0, 0])  # 初始方向向量

    # Calculate the rotation matrix with Euler Angles
    rotation_matrix = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    # Return the direction vector
    rotated_vector = np.dot(rotation_matrix, direction_vector)
    return rotated_vector


test_obj = [folder.split('/')[0] for folder in open('vis_file.txt')]
k = 0
for i, obj in enumerate(test_obj):
    traj_gt = []
    pose_file = pd.read_excel('/home/slj/EndoTraj/NEPose-main/data/{}/traj.xlsx'.format(obj), header=None)
    for pose in pose_file.values:
        if str(pose[4]) != 'OK':
            continue
        traj_gt.append(pose[9:12])
    length = 0
    for i in range(1, len(traj_gt)):
        length += cal_distance(traj_gt[i], traj_gt[i - 1])

    d_gt = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono/results_{}/directions_gt.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_gt.append(pos.split(' '))
    d_pred_offset = []
    d_pred_endoslam = []
    d_pred_mono = []
    d_pred_simcol = []
    d_pred_ours = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_offsetnet/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_offset.append(pos.split(' '))
    with open(
            '/home/slj/EndoTraj/NEPose-main/results/results_endoslam/results_{}/directions_pred.txt'.format(obj),
            'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_endoslam.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_mono.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_simcol/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_simcol.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ours/results_{}/directions_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            d_pred_ours.append(pos.split(' '))
    traj_gt = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono/results_{}/traj_gt.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_gt.append(pos.split(' '))
    traj_pred_offset = []
    traj_pred_endoslam = []
    traj_pred_mono = []
    traj_pred_simcol = []
    traj_pred_ours = []
    with open('/home/slj/EndoTraj/NEPose-main/results/results_offsetnet/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_offset.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_endoslam/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_endoslam.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_mono/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_mono.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_simcol/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_simcol.append(pos.split(' '))
    with open('/home/slj/EndoTraj/NEPose-main/results/results_ours/results_{}/traj_pred.txt'.format(obj),
              'r') as file:
        lines = file.readlines()
        for line in lines:
            pos = line.strip()
            traj_pred_ours.append(pos.split(' '))

    x_gt = [0]
    y_gt = [0]
    z_gt = [0]
    x_pred_offset = [0]
    y_pred_offset = [0]
    z_pred_offset = [0]
    x_pred_endoslam = [0]
    y_pred_endoslam = [0]
    z_pred_endoslam = [0]
    x_pred_mono = [0]
    y_pred_mono = [0]
    z_pred_mono = [0]
    x_pred_simcol = [0]
    y_pred_simcol = [0]
    z_pred_simcol = [0]
    x_pred_ours = [0]
    y_pred_ours = [0]
    z_pred_ours = [0]
    for i in range(1, len(traj_gt)):
        x_gt.append((float(traj_gt[i][0])-float(traj_gt[0][0])))
        y_gt.append((float(traj_gt[i][1])-float(traj_gt[0][1])))
        z_gt.append((float(traj_gt[i][2])-float(traj_gt[0][2])))
        x_pred_offset.append((float(traj_pred_offset[i][0])-float(traj_gt[0][0])))
        y_pred_offset.append((float(traj_pred_offset[i][1])-float(traj_gt[0][1])))
        z_pred_offset.append((float(traj_pred_offset[i][2])-float(traj_gt[0][2])))
        x_pred_endoslam.append((float(traj_pred_endoslam[i][0])-float(traj_gt[0][0])))
        y_pred_endoslam.append((float(traj_pred_endoslam[i][1])-float(traj_gt[0][1])))
        z_pred_endoslam.append((float(traj_pred_endoslam[i][2])-float(traj_gt[0][2])))
        x_pred_mono.append((float(traj_pred_mono[i][0])-float(traj_gt[0][0])))
        y_pred_mono.append((float(traj_pred_mono[i][1])-float(traj_gt[0][1])))
        z_pred_mono.append((float(traj_pred_mono[i][2])-float(traj_gt[0][2])))
        x_pred_simcol.append((float(traj_pred_simcol[i][0])-float(traj_gt[0][0])))
        y_pred_simcol.append((float(traj_pred_simcol[i][1])-float(traj_gt[0][1])))
        z_pred_simcol.append((float(traj_pred_simcol[i][2])-float(traj_gt[0][2])))
        x_pred_ours.append((float(traj_pred_ours[i][0])-float(traj_gt[0][0])))
        y_pred_ours.append((float(traj_pred_ours[i][1])-float(traj_gt[0][1])))
        z_pred_ours.append((float(traj_pred_ours[i][2])-float(traj_gt[0][2])))

    # Visualization of the absolute trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', labelsize=14)
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    ax.set_title('The whole length: {:.2f}cm'.format(length / 10), fontsize=20)
    ax.plot(x_gt, y_gt, z_gt, label='GT', color='k')
    ax.plot(x_pred_offset, y_pred_offset, z_pred_offset, label='OffsetNet', color='c', linewidth=2)
    ax.plot(x_pred_endoslam, y_pred_endoslam, z_pred_endoslam, label='Endo-SfM', color='g', linewidth=2)
    ax.plot(x_pred_mono, y_pred_mono, z_pred_mono, label='Monodepth2', color='b', linewidth=2)
    ax.plot(x_pred_simcol, y_pred_simcol, z_pred_simcol, label='SimCol', color='y', linewidth=2)
    ax.plot(x_pred_ours, y_pred_ours, z_pred_ours, label='Ours', color='r', linewidth=2)
    ax.set_xlabel('X axis/mm', labelpad=12, fontsize=16)
    ax.set_ylabel('Y axis/mm', labelpad=12, fontsize=16)
    ax.set_zlabel('Z axis/mm', labelpad=12, fontsize=16)
    ax.legend(fontsize=16)
    plt.show()

    # # Visualization of the relative pose
    # for i in range(0, len(x_gt)-1, 2):
    #     j = (i+1) // 2
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.tick_params(axis='both', labelsize=14)
    #     plt.rcParams['font.family'] = ['serif']
    #     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    #
    #     # Draw the translation
    #     ax.plot(x_gt[i:i+2], y_gt[i:i+2], z_gt[i:i+2], '--', color='k', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_offset[i+1]], [y_gt[i]]+[y_pred_offset[i+1]], [z_gt[i]]+[z_pred_offset[i+1]], '--', label='OffsetNet', color='c', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_endoslam[i+1]], [y_gt[i]]+[y_pred_endoslam[i+1]], [z_gt[i]]+[z_pred_endoslam[i+1]], '--', label='Endo-SfM', color='g', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_mono[i+1]], [y_gt[i]]+[y_pred_mono[i+1]], [z_gt[i]]+[z_pred_mono[i+1]], '--', label='Monodepth2', color='b', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_simcol[i+1]], [y_gt[i]]+[y_pred_simcol[i+1]], [z_gt[i]]+[z_pred_simcol[i+1]], '--', label='SimCol', color='y', linewidth=2)
    #     ax.plot([x_gt[i]]+[x_pred_ours[i+1]], [y_gt[i]]+[y_pred_ours[i+1]], [z_gt[i]]+[z_pred_ours[i+1]], '--', label='Ours', color='r', linewidth=2)
    #     d_0 = caldir(float(d_gt[j+1][0]), float(d_gt[j+1][1]), float(d_gt[j+1][2]))
    #     d_offset = caldir(float(d_pred_offset[j+1][0]), float(d_pred_offset[j+1][1]), float(d_pred_offset[j+1][2]))
    #     d_mono = caldir(float(d_pred_mono[j+1][0]), float(d_pred_mono[j+1][1]), float(d_pred_mono[j+1][2]))
    #     d_endoslam = caldir(float(d_pred_endoslam[j+1][0]), float(d_pred_endoslam[j+1][1]), float(d_pred_endoslam[j+1][2]))
    #     d_simcol = caldir(float(d_pred_simcol[j+1][0]), float(d_pred_simcol[j+1][1]), float(d_pred_simcol[j+1][2]))
    #     d_ours = caldir(float(d_pred_ours[j+1][0]), float(d_pred_ours[j+1][1]), float(d_pred_ours[j+1][2]))
    #
    #     # Draw the directions
    #     ax.quiver(x_gt[i+1], y_gt[i+1], z_gt[i+1], d_0[0], d_0[1], d_0[2], color='k', label='Direction of GT', linewidth=2, length=0.05)
    #     ax.quiver(x_pred_offset[i+1], y_pred_offset[i+1], z_pred_offset[i+1], d_offset[0], d_offset[1], d_offset[2], color='c', label='Direction of OffsetNet', linewidth=2, length=0.05)
    #     ax.quiver(x_pred_mono[i+1], y_pred_mono[i+1], z_pred_mono[i+1], d_mono[0], d_mono[1], d_mono[2], color='b', label='Direction of MonoDepth2', linewidth=2, length=0.05)
    #     ax.quiver(x_pred_endoslam[i+1], y_pred_endoslam[i+1], z_pred_endoslam[i+1], d_endoslam[0], d_endoslam[1], d_endoslam[2], color='g', label='Direction of Endo-SfM', linewidth=2, length=0.05)
    #     ax.quiver(x_pred_simcol[i+1], y_pred_simcol[i+1], z_pred_simcol[i+1], d_simcol[0], d_simcol[1], d_simcol[2], color='y', label='Direction of SimCol', linewidth=2, length=0.05)
    #     ax.quiver(x_pred_ours[i+1], y_pred_ours[i+1], z_pred_ours[i+1], d_ours[0], d_ours[1], d_ours[2], color='r', label='Direction of Ours', linewidth=2, length=0.05)
    #     ax.set_xlabel('X axis/mm', labelpad=12, fontsize=16)
    #     ax.set_ylabel('Y axis/mm', labelpad=12, fontsize=16)
    #     ax.set_zlabel('Z axis/mm', labelpad=12, fontsize=16)
    #     ax.legend(fontsize=16)
    #     plt.show()



