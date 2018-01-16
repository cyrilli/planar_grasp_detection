# planar_grasp_detection
A ROS package that detects grasp rectangles from rgb image and converts that rectangle to gripper pose using depth image

The package contains two ROS services, GraspDetection and GraspConversion. For the grasp detection server, I adapted the code from paper [Supervision via Competition: Robot Adversaries for Learning Tasks](https://arxiv.org/pdf/1610.01685v1.pdf), which basically takes rgb image as input, gets patches of image (the size is determined by the parameter gscale, the ratio of grasp rectangle w.r.t image), resizes patches into shape 224\*224\*3, feeds them into convolutional neural network and get scores for each rotation angle for the patch. This ROS service provides the 2D location (center of rectangle) and a rotation angle.

The second service takes results from grasp detection and depth image as inputs, and converts to gripper pose using the procedure described in the paper Deep Learning for Detecting Robotic Grasps. Note that I am using a robotiq 85 gripper, so people using a different gripper should change the FINGER_LENGTH accordingly.

Below is screen shot of how it looks how in my application. The third window on the right shows the detected grasp rectangles, and the arrows in the main window are the converted grasp poses.

![demo_img](https://github.com/cyrilli/planar_grasp_detection/blob/master/img/demo_image.png?raw=true)
