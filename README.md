# NEPose: An Endoscopic Video Dataset and a Deep Learning-based Method for Nasal Endoscopy Pose Estimation
Welcome to the Project Page! :grin:

In this project, :star: the first dataset for pose estimation of nasal endoscope :star:, which consists of endoscopic videos from real objects and corresponding pose of endoscope in each frame, is proposed. Meanwhile, :star: a deep learning-based model :star:, including feature extractors and a novel pose decoder, is also proposed to estimated the pose of the endoscope from the monocular endoscopic images.

- :heavy_check_mark: [The Exhibition of Our Dataset](#visualization-of-our-dataset) is available here!
- :heavy_check_mark: [The Exhibition of the Results](#visualization-of-our-estimated-trajectories) is available here!
- :black_square_button: The whole code and dataset will be public when the paper :page_with_curl: is published.

## Visualization of Our Dataset
The visualization of our data includes:

- The endoscopic video obtained with a nasal endoscope: :triangular_flag_on_post: `/data/{Obj}/Frames`.
- The corresponding trajectory of the endoscope obtained with a optical tracking system: :triangular_flag_on_post: `/data/{Obj}/traj.xlsx`.

We select 10 sets of data from the dataset to show here:

<img src="/vis_data/1.gif" width="400px"> <img src="/vis_data/2.gif" width="400px">

<img src="/vis_data/3.gif" width="400px"> <img src="/vis_data/4.gif" width="400px">

<img src="/vis_data/5.gif" width="400px"> <img src="/vis_data/6.gif" width="400px">

<img src="/vis_data/7.gif" width="400px"> <img src="/vis_data/8.gif" width="400px">

<img src="/vis_data/9.gif" width="400px"> <img src="/vis_data/10.gif" width="400px">

## Visualization of Our Estimated Trajectories
We utilize the absolute pose of each frame from the video to generate the trajectory of the endoscope, which canbe used as qualitative evaluation of methods.

<img src="/vis_data/res1.gif" width="700px"><img src="/vis_data/legend.png" width="100px">

<img src="/vis_data/res2.gif" width="700px"><img src="/vis_data/legend.png" width="100px">

<img src="/vis_data/res3.gif" width="700px"><img src="/vis_data/legend.png" width="100px">
