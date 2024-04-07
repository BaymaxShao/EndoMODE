'''
This is the code for the visualization of our data.
'''


import matplotlib.pyplot as plt
from matplotlib.image import imread
from path import Path
import pandas as pd
import os

data_root = './data'
root = Path(data_root)
objs = [root/folder.split('/')[0] for folder in open('data_list.txt')]
for i, obj in enumerate(objs):
    os.mkdir('./data_vis/{}'.format(str(obj)[-3:]))
    positions = []
    frames = obj / 'Frames'
    imgs = sorted(frames.files('*.jpg'))
    pose_name = Path('traj.xlsx')
    pose_file = pd.read_excel(obj / pose_name, header=None)
    for pose in pose_file.values:
        positions.append(pose[9:12])
    # 初始化图形和轴
    fig = plt.figure()
    traj = fig.add_subplot(122, projection='3d')
    fs = fig.add_subplot(121)
    traj.set_xlabel('X/mm')
    traj.set_ylabel('Y/mm')
    traj.set_zlabel('Z/mm')

    # 初始化曲线对象
    line, = traj.plot([], [], [], lw=2)

    for j in range(0, len(positions), 4):
        x, y, z = [x[0]-positions[0][0] for x in positions[0:j+1:4]], [x[1]-positions[0][1] for x in positions[0:j+1:4]], [x[2]-positions[0][2] for x in positions[0:j+1:4]]
        traj.plot(x, y, z, lw=2, color='k')
        img = imread(str(imgs[j]))
        fs.imshow(img)
        fs.axis('off')
        print('{}/{} ok!'.format(j//4+1, len(positions)//4+1))
        plt.savefig('./data_vis/{}/{}.png'.format(str(obj)[-3:], j//4+1))

    print('data {} finished'.format(str(obj)[-3:]))

