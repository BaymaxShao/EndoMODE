# NEPose: An Endoscopic Video Dataset and a Deep Learning-based Method for Nasal Endoscopy Pose Estimation
The first dataset for pose estimation of nasal endoscope, which consists of endoscopic videos from real objects and corresponding pose of endoscope in each frame, is proposed. Meanwhile, a deep learning-based model, including feature extractors and a novel pose decoder, is also proposed to estimated the pose of the endoscope from the monocular endoscopic images.

- The visualizations of some data in the dataset are available here!
- The whole code and dataset will be public when the work is accepted.

The Project currently contains:
- [The Exhibition of Our Dataset](#visualization-of-our-dataset)
- [The Exhibition of the Results](#visualization-of-our-estimated-trajectories)

## Visualization of Our Dataset
The visualization of our data includes:

- The endoscopic video obtained with a nasal endoscope: `/data/{Obj}/Frames`.
- The corresponding trajectory of the endoscope obtained with a optical tracking system: `/data/{Obj}/traj.xlsx`.

We select 10 sets of data from the dataset to show here:

<img src="/vis_data/1.gif" width="400px"> <img src="/vis_data/2.gif" width="400px">

<img src="/vis_data/3.gif" width="400px"> <img src="/vis_data/4.gif" width="400px">

<img src="/vis_data/5.gif" width="400px"> <img src="/vis_data/6.gif" width="400px">

<img src="/vis_data/7.gif" width="400px"> <img src="/vis_data/8.gif" width="400px">

<img src="/vis_data/9.gif" width="400px"> <img src="/vis_data/10.gif" width="400px">

## Visualization of Our Estimated Trajectories
