# High-Precision 6-DoF Grasp Detection in Cluttered Scenes Based on Network Optimization and Pose Propagation

High precision grasp pose detection is an essential but challenging task in robotic manipulation. Most of the current methods for grasp detection either highly rely on the geometry information of the objects or generate feasible grasp poses within restricted configurations. In this letter, a grasp pose detection framework is proposed that generates a rich set of 6-DoF grasp poses with high precision. First, a Multi-radius Cylinder Sampling and Feature Fusion module (MCS-FF) is designed to enhance local geometric representation. Second, an optimized grasp operation head is developed to further estimate grasp parameters. Finally, a grasp pose propagation algorithm is proposed, which effectively extends grasp poses from a restricted configuration to a larger configuration. Experiments on the large-scale benchmark, GraspNet-1Billion, show that the proposed method outperforms existing methods, improving the average precision by 8.61%. The real-world experiments further demonstrate the effectiveness of the proposed method in cluttered environments.

The structure of the framework
-----

![image](https://github.com/WenJunTang2000/6-DoF-Grasp/blob/main/img/structure.png)

Results
-----
Evaluation results on Realsense camera:

The visualization of detected grasp poses. 

![image](https://github.com/WenJunTang2000/6-DoF-Grasp/blob/main/img/vis.png)
The visualization of detected grasp poses. Grippers in red are well grasps while those in purler and blue are collision and bad ones respectively. The first row shows the results of the baseline method, the second row shows the results of our method, and the third row displays the improvement in AP.

Video
-----

<iframe 
src="img/video.mp4" 
scrolling="no" 
border="0" 
frameborder="no" 
framespacing="0" 
allowfullscreen="true" 
height=600 
width=800> 
</iframe>
