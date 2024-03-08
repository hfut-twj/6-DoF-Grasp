# High-Precision 6-DoF Grasp Detection in Cluttered Scenes Based on Network Optimization and Pose Propagation

Results
-----
Evaluation results based on the GraspNet-1Billion benchmark captured by RealSense:

|          |        | Seen             |                  |        | Similar          |                  |        | Novel            |                  | 
|:--------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|
|          | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| Ours     | 74.31  | 85.23            | 70.04            | 64.72  | 77.48            | 56.88            | 26.66  | 33.18            | 14.41             |
| Ours + CD| 75.39  | 86.75            | 70.60            | 65.75  | 78.82            | 57.52            | 27.38  | 34.17            | 14.56             |


The visualization of detected grasp poses:

![image](https://github.com/WenJunTang2000/6-DoF-Grasp/blob/main/img/vis.png)
Grippers in red are well grasps while those in purler and blue are collision and bad ones respectively. The first row shows the results of the baseline method, the second row shows the results of our method, and the third row displays the improvement in AP.

Video
-----
The real-world grasping experiments:
[![Alt text](https://github.com/hfut-twj/6-DoF-Grasp/blob/main/img/cover.png)](https://www.youtube.com/watch?v=uB1haZOncH4)

The visualization of the sampled grasp poses:
![image](https://github.com/hfut-twj/6-DoF-Grasp/blob/main/img/vis_exp.png)

Acknowledgment
-----
My code is mainly based on Graspnet-baseline https://github.com/graspnet/graspnet-baseline and https://github.com/rhett-chen/graspness_implementation.

