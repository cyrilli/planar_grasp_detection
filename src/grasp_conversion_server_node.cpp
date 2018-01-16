/*******************************************************************************
 * Copyright (C) 2017 Harbin Institute of Technology
 *
 * This file is part of the Pick and Place Project for my Master's thesis
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Chuanhao Li

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/pinhole_camera_model.h>
#include <geometry_msgs/PoseStamped.h>
#include <moveit_msgs/Grasp.h>
#include <sensor_msgs/CameraInfo.h>
#include "grasp_detection/GraspPoseConverter.h"
#include "grasp_detection/GraspConversion.h"  //the service
#include <iostream>
#include <math.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>

#define PI 3.14159265
#define FINGER_LENGTH 0.1

bool convert_grasp(grasp_detection::GraspConversion::Request  &req,
                   grasp_detection::GraspConversion::Response &res)
{
  /*
  # request
  sensor_msgs/Image depth_image
  std_msgs/Int32MultiArray best_grasps_h
  std_msgs/Int32MultiArray best_grasps_w
  std_msgs/Float32MultiArray best_grasps_angle
  sensor_msgs/CameraInfo info
  */
  tf::TransformListener listener;
  image_geometry::PinholeCameraModel cam_model;
  cam_model.fromCameraInfo(req.info); // Build the camera model from CameraInfo, then use it to get cx, cy, fx, fy
  cv::Mat depth_img = cv_bridge::toCvCopy(req.depth_image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  GraspPoseConverter grasp_converter(depth_img, 1.0, cam_model.cx(), cam_model.cy(), cam_model.fx(), cam_model.fy());
  int i = 0;
  for (std::vector<int>::const_iterator it = req.best_grasps_h.data.begin();it != req.best_grasps_h.data.end(); ++it){
    grasp_converter.convertTo3DGrasp(req.best_grasps_h.data[i], req.best_grasps_w.data[i], req.best_grasps_angle.data[i]);

    //After get the grasp point and rotation vector, and rotation angle we need to convert it to quaternion
    float qw = cos((grasp_converter.rotAng)*PI/(2*180));
    float qx = grasp_converter.rotVec[0]*sin((grasp_converter.rotAng)*PI/(2*180));
    float qy = grasp_converter.rotVec[1]*sin((grasp_converter.rotAng)*PI/(2*180));
    float qz = grasp_converter.rotVec[2]*sin((grasp_converter.rotAng)*PI/(2*180));
    //Note these are in the camera frame, so we still need to convert it to the world frame
    geometry_msgs::PoseStamped pose_camera;
    geometry_msgs::PoseStamped pose_world;

    pose_camera.pose.position.x = grasp_converter.graspPoint3D.x + FINGER_LENGTH * grasp_converter.rotVec[0];
    pose_camera.pose.position.y = grasp_converter.graspPoint3D.y + FINGER_LENGTH * grasp_converter.rotVec[1];
    pose_camera.pose.position.z = grasp_converter.graspPoint3D.z + FINGER_LENGTH * grasp_converter.rotVec[2];  // some distance for the finger tip
    pose_camera.pose.orientation.x = qx;
    pose_camera.pose.orientation.y = qy;
    pose_camera.pose.orientation.z = qz;
    pose_camera.pose.orientation.w = qw;
    pose_camera.header.stamp = ros::Time(0);
    pose_camera.header.frame_id = req.info.header.frame_id; //'/rgbd_camera_optical_frame'

    try{
      listener.waitForTransform("/world", req.info.header.frame_id, ros::Time(0), ros::Duration(1.0));
      listener.transformPose("/world", pose_camera, pose_world);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
    }
    res.grasp_poses.push_back(pose_world);
    i++;
  }
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "grasp_conversion_server_node");
  ros::NodeHandle n;
  ros::ServiceServer service = n.advertiseService("grasp_conversion", convert_grasp);
  ROS_INFO("Ready to convert grasp.");
  ros::spin();
  return 0;
}