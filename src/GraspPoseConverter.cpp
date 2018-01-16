/*******************************************************************************
 * Copyright (C) 2017 Harbin Institute of Technology
 *
 * This file is part of the Pick and Place Project for my Master's thesis
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Chuanhao Li

#include "grasp_detection/GraspPoseConverter.h"

GraspPoseConverter::GraspPoseConverter(const cv::Mat &input_img, double c_factor = 1000.0, double cx = 320.5, double cy = 240.5,
                                       double fx = 577.2959393832757, double fy = 577.2959393832757):depth_img(input_img){
  if(depth_img.empty()){
    printf("Converter instantiation failed! Depth image is empty!\n");
  }
  else{
    std::cout<<"======Successfully instantiate converter!======\n"<<std::endl;
    //cv::imwrite("/home/lichuanhao/depth_img.png", depth_img);
    inRectMask = cv::Mat::zeros(depth_img.size(), CV_8UC1);
    nonZeroMask = cv::Mat::zeros(depth_img.size(), CV_8UC1);
    //std::cout<<"inRectMask before size"<<inRectMask.size()<<std::endl;

    img_height = depth_img.rows;
    img_width = depth_img.cols;
    camera_factor = c_factor;
    camera_cx = cx;
    camera_cy = cy;
    camera_fx = fx;
    camera_fy = fy;
  }
}

GraspPoseConverter::~GraspPoseConverter(){}

void GraspPoseConverter::convertTo3DGrasp(const int &h, const int &w, const float &angle){
  /*
  row == height == Point.y
  col == width == Point.x
  */
  graspPoint2D.x = w;
  graspPoint2D.y = h;
  rotAng = -angle;

  robotNomrAroundClosestPt();  // compute rotVec and minD
  convertImagePointTo3DGraspPoint();  // use graspPoint2D and minD to compute graspPoint3D
}

void GraspPoseConverter::robotNomrAroundClosestPt(){
  // First get the region around graspPoint2D
  float radius = 5;
  cv::circle(inRectMask, graspPoint2D, radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
  cv::threshold(depth_img, nonZeroMask, 0, 255, cv::THRESH_BINARY);   //nonZeroMask type is not type0 after this
  nonZeroMask.convertTo(nonZeroMask, CV_8UC1);

  // get minimum depth: minD in the region, and its location: minDepthPoint2D
  cv::Mat inRectNonZeroMask = cv::Mat::zeros(depth_img.size(), CV_8UC1);
  cv::bitwise_and(inRectMask, nonZeroMask, inRectNonZeroMask);

  cv::minMaxLoc(depth_img, &minD, NULL, &minDepthPoint2D, NULL, inRectNonZeroMask);    // use inersection between two masks to get non zero minimum value
  ave3DNormAroundP(); // Compute the average surface normal around minDepthPoint2D: rotVec
}

void GraspPoseConverter::ave3DNormAroundP(){
  smoothDepthForNorm();
  depthPatchAroundP(); //get the region around minDepthPoint2D, save it in minDepthPointRegion
  //thresholdFromCenterValue();   // The problem is I won't be able to use this to avoid using some bad surface normals for averaging after converting to point cloud
  getSurfNorm3D();
}

void GraspPoseConverter::smoothDepthForNorm(){
  int KERNEL_LENGTH = 5;
  cv::blur(depth_img,depth_img_smooth,cv::Size(KERNEL_LENGTH,KERNEL_LENGTH));
}

void GraspPoseConverter::depthPatchAroundP(){
  int PATCH_WD = 5;  //5
  minDepthPointRegion = depth_img_smooth(cv::Range(std::max(0,minDepthPoint2D.y-PATCH_WD), std::min(minDepthPoint2D.y+PATCH_WD, img_height)),
                                         cv::Range(std::max(0, minDepthPoint2D.x-PATCH_WD), std::min(minDepthPoint2D.x+PATCH_WD, img_width)));
}

void GraspPoseConverter::thresholdFromCenterValue(){
  /*
    Compute a mask of pixels which are more than some threshold's worth from the 
    value of the center pixel of the given image
  */
  int DIST_THRESH = 30;
  int centVal = int(minDepthPointRegion.at<float>(std::floor(minDepthPointRegion.rows/2),std::floor(minDepthPointRegion.cols/2)));
  cv::threshold(minDepthPointRegion, centDepthValThresholdMask, centVal+DIST_THRESH, 255, cv::THRESH_BINARY_INV); 
}

void GraspPoseConverter::getSurfNorm3D(){
  /*
    First convert all points in the image to 3D coordinates,
    and then compute their surface normals.
  */
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  // loop over the whole depth map
  for (int m = 0; m < minDepthPointRegion.rows; m++){
    for (int n = 0; n < minDepthPointRegion.cols; n++){
      // get the value at (m,n) in the depth image
      //double d = minDepthPointRegion.ptr<double>(m)[n]; // the output values are sooo big!!
      double d = (double) minDepthPointRegion.at<float>(m, n);
      // if d has no value, skip
      if (d == 0){
        continue;
      }
      if (d-minD > 0.05){
        continue;
      }
      // if d has value, add a point
      pcl::PointXYZ p;

      // compute the 3D coordinate of this point
      p.z = d / camera_factor;
      p.x = (n - camera_cx) * p.z / camera_fx;
      p.y = (m - camera_cy) * p.z / camera_fy;
      // add p into cloud
      cloud->points.push_back(p);
    }
  }
  // set some params
  cloud->height = 1;
  cloud->width = cloud->points.size();
  cloud->is_dense = false;
  //std::cout<<"cloud contains num of points: "<<cloud->points.size()<<std::endl;
  //pcl::io::savePCDFileASCII("/home/lichuanhao/minDepthPointRegion.pcd", *cloud);

  // Create the normal estimation class, and pass the input dataset to it
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch (0.03);

  // Compute the features
  ne.compute (*cloud_normals);

  float sum_normal_x, sum_normal_y, sum_normal_z;
  // iterate over each points in cloud_normals
  for(pcl::PointCloud<pcl::Normal>::iterator it = cloud_normals->begin(); it!=cloud_normals->end();it++){
    sum_normal_x += it->normal_x;
    sum_normal_y += it->normal_y;
    sum_normal_z += it->normal_z;
  }
  float point_num = (float) cloud_normals->points.size();
  rotVec[0] = sum_normal_x / point_num;
  rotVec[1] = sum_normal_y / point_num;
  rotVec[2] = sum_normal_z / point_num;
  rotVec = cv::normalize(rotVec);
}

void GraspPoseConverter::convertImagePointTo3DGraspPoint(){
    // compute the 3D coordinate of this point
    graspPoint3D.z = minD / camera_factor;
    graspPoint3D.x = (minDepthPoint2D.x - camera_cx) * graspPoint3D.z / camera_fx;
    graspPoint3D.y = (minDepthPoint2D.y - camera_cy) * graspPoint3D.z / camera_fy;
}