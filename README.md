# planar_grasp_detection
A ROS package that detects grasp rectangles from rgbd image and converts that rectangle to gripper pose using depth image.

The project is still under progress. Currently I adopted tons of code from many other sources. Three main components are written as ROS services and called from another python script to make it easy to tinker with. Not sure if this wastes too much time on sending and receiving messages.

The three ROS services are: 
1. GraspSampling.srv
2. GraspDetection.srv
3. GraspConversion.srv 

GraspSampling code is adopted from [gqcnn](https://github.com/BerkeleyAutomation/gqcnn) to generate candidate antipodal grasps from a depth image. Then I segment a square image patch from each antipodal grasp (size of the image patch is 1.2 times of the gripper width), and feed this to the grasp detection server where a convolutional neural network is used to make predictions based on this patch. Therefore, this step gives us image patches (potentially of different size since the gripper would look smaller in the image if it is further) at the location of antipodal grasps.

For the grasp detection server, I adapted the code from paper [Supervision via Competition: Robot Adversaries for Learning Tasks](https://arxiv.org/pdf/1610.01685v1.pdf), which takes rgb image patch as input. Note that all patches are resized into shape 224\*224\*3, and then fed into convolutional neural network.  The neural net is trained to produce probability of a successful grasp for each rotation angle (0, 10, ..., 170, that is 18 angles in total) on the given patch.

Then I applied a weight matrix on the probability to combine the predictions on angles and the computed antipodal angle based on the angle difference.

The third service takes results from grasp detection and depth image as inputs, and converts to gripper pose using the procedure described in the paper Deep Learning for Detecting Robotic Grasps. Note that I am using a robotiq 85 gripper, so people using a different gripper should change the FINGER_LENGTH accordingly.

Below is screen shot of how it looks in my application. The third window on the right shows the detected grasp rectangles, and the arrows in the main window are the converted grasp poses.

![demo_img](https://github.com/cyrilli/planar_grasp_detection/blob/master/img/demo_image.png?raw=true)

The following is a video demo using this ROS packge to grasp objects from a table in Gazebo simulation.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/qlhEtHCVZsw/0.jpg)](http://www.youtube.com/watch?v=qlhEtHCVZsw "Robot Grasp Pose Detection Demo")
