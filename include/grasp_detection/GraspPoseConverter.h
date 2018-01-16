/*******************************************************************************
 * Copyright (C) 2017 Harbin Institute of Technology
 *
 * This file is part of the Pick and Place Project for my Master's thesis
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Chuanhao Li

#ifndef GRASPPOSECONVERTER_H
#define GRASPPOSECONVERTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <math.h>
#include <iostream>

using std::cout;

class GraspPoseConverter {
public:
  GraspPoseConverter(const cv::Mat &input_img, double c_factor, double cx, double cy, double fx, double fy);
  ~GraspPoseConverter();

  cv::Point3f graspPoint3D;
  cv::Vec3f rotVec;
  float rotAng;

  void convertTo3DGrasp(const int &h, const int &w, const float &angle);

private:
  void convertTo3DGrasp();
  void robotNomrAroundClosestPt();
  void ave3DNormAroundP();
  void smoothDepthForNorm();    //
  void depthPatchAroundP();
  void thresholdFromCenterValue(); // first convert to 3D space then compute surface normals
  void getSurfNorm3D();
  void convertImagePointTo3DGraspPoint();

  cv::Point2i graspPoint2D; //store the 2D grasp point
  cv::Point2i minDepthPoint2D; //the point with minimum depth in the region around grasp point
  double minD; //mimimun depth in the region around grasp point
  int img_height;
  int img_width;
  cv::Mat depth_img;          // Depth Image
  cv::Mat depth_img_smooth;
  cv::Mat inRectMask;      // mask of the ROI region centered at grasp point
  cv::Mat nonZeroMask;
  // compare the depth value of each point in depth_img_smooth, and get the mask of those who
  // has depth value that is smaller than centVal + thresh
  cv::Mat centDepthValThresholdMask;  
  cv::Mat minDepthPointRegion;  // the ROI region centered at minDepthPoint2D
  
  double camera_factor;
  double camera_cx;
  double camera_cy;
  double camera_fx;
  double camera_fy;
};

#endif // GRASPPOSECONVERTER_H