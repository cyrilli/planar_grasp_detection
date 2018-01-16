#!/usr/bin/env python
"""
Copyright (C) 2017 Harbin Institute of Technology
 
This file is part of the Pick and Place Project for my Master's thesis

All Rights Reserved.

Author: Chuanhao Li

This node is the server for the service GraspDetection, which takes a RGB image as input and detects grasp pose as
bounding boxes.
"""


import rospy
import roslib
import rospkg
roslib.load_manifest('grasp_detection')

import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from grasp_detection.patch_based_grasp_detection.grasp_learner import grasp_obj
from grasp_detection.patch_based_grasp_detection.grasp_predictor import Predictors
from grasp_detection.srv import GraspDetection, GraspDetectionResponse
from sensor_msgs.msg import Image

class GraspDetector():
    def __init__(self):
        self._cv_bridge = CvBridge()
        rospack = rospkg.RosPack()
        model_path = rospack.get_path('grasp_detection')+'/models/patch_based_grasp_detection/Grasp_model'
        nsamples = 500
        self.nbest = 10
        max_batchsize = 128
        gpu_id = 0 # set to -1 when using cpu

        ## Set up model
        if nsamples > max_batchsize:
            self.batchsize = max_batchsize
            self.nbatches = int(nsamples / max_batchsize) + 1
        else:
            self.batchsize = nsamples
            self.nbatches = 1
        
        print('Loading grasp model')
        self.G = grasp_obj(model_path, gpu_id)
        self.G.BATCH_SIZE = self.batchsize
        self.G.test_init()

        self._visualization = rospy.get_param('~visualization', True)
        self.vis_pub = rospy.Publisher('/grasp_detection/visualization', Image, queue_size=1)
        self.s = rospy.Service('grasp_detection', GraspDetection, self.handle_grasp_detection)
        #self.G.test_close()
        rospy.loginfo("Waiting for request!")

    def handle_grasp_detection(self, req):
        """
        After receiving request from client for GraspDetection service, call GraspDetector to
        find best grasps in the RGB image.
        :param req: GraspDetection service request, including color_image and gscale
        :return: GraspDetection service response, including h, w, and angle of best grasps
        """
        rospy.loginfo("Request received!")
        I = np.array(self._cv_bridge.imgmsg_to_cv2(req.color_image, "bgr8"))
        imsize = max(I.shape[:2])
        gsize = int(req.gscale.data * imsize)  # Size of grasp patch
        P = Predictors(I, self.G)

        fc8_predictions = []
        patch_Hs = []
        patch_Ws = []

        st_time = time.time()
        for _ in range(self.nbatches):
            P.graspNet_grasp(patch_size=gsize, num_samples=self.batchsize);
            fc8_predictions.append(P.fc8_norm_vals)
            patch_Hs.append(P.patch_hs)
            patch_Ws.append(P.patch_ws)

        fc8_predictions = np.concatenate(fc8_predictions)
        patch_Hs = np.concatenate(patch_Hs)
        patch_Ws = np.concatenate(patch_Ws)

        largest_idxs = self.k_largest_index_argsort(fc8_predictions, self.nbest)
        best_patch_Hs = []
        best_patch_Ws = []
        best_patch_Angles = []
        for idx in largest_idxs:
            if self._visualization:
                I = self.drawRectangle(I, patch_Hs[idx[0]], patch_Ws[idx[0]], idx[1], gsize)
            best_patch_Hs.append(patch_Hs[idx[0]])
            best_patch_Ws.append(patch_Ws[idx[0]])
            grasp_angle = (idx[1] * (np.pi / 18) - np.pi / 2) * 180 / np.pi   # convert to degree
            best_patch_Angles.append(grasp_angle)

        rospy.loginfo('Time taken: {}s'.format(time.time() - st_time))
        resp = GraspDetectionResponse()
        resp.best_grasps_h = Int32MultiArray(data = best_patch_Hs)
        resp.best_grasps_w = Int32MultiArray(data = best_patch_Ws)
        resp.best_grasps_angle = Float32MultiArray(data = best_patch_Angles)
        if self._visualization:
            result_msg = self._cv_bridge.cv2_to_imgmsg(I, "bgr8")
            self.vis_pub.publish(result_msg)
        #cv2.imwrite("/home/lichuanhao/bbox.png", I)
        return resp

    def k_largest_index_argpartition(self, a, k):
        """
        Find the index of the largest value in array.
        The returned index is not in order, but this is computationally cheaper.
        :param a: a 2D array
        :param k: k biggest values
        :return: index of k biggest value in array a
        """
        idx = np.argpartition(a.ravel(), a.size-k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))

    def k_largest_index_argsort(self, a, k):
        """
        Find the index of the largest value in array.
        The returned index is in order.
        :param a: a 2D array
        :param k: k biggest values
        :return: index of k biggest value in array a
        """
        idx = np.argsort(a.ravel())[:-k-1:-1]
        return np.column_stack(np.unravel_index(idx, a.shape))

    def drawRectangle(self, I, h, w, t, gsize=300):
        I_temp = I
        grasp_l = gsize / 2.5
        grasp_w = gsize / 5.0
        grasp_angle = t * (np.pi / 18) - np.pi / 2
        points = np.array([[-grasp_l, -grasp_w],
                           [grasp_l, -grasp_w],
                           [grasp_l, grasp_w],
                           [-grasp_l, grasp_w]])
        R = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                      [np.sin(grasp_angle), np.cos(grasp_angle)]])
        rot_points = np.dot(R, points.transpose()).transpose()
        im_points = rot_points + np.array([w, h])
        cv2.line(I_temp, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0, 255, 0),
                 thickness=5)
        cv2.line(I_temp, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0, 0, 255),
                 thickness=5)
        cv2.line(I_temp, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0, 255, 0),
                 thickness=5)
        cv2.line(I_temp, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0, 0, 255),
                 thickness=5)
        return I_temp

    def main(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node('patch_based_grasp_detection_node', anonymous = True)
    gd = GraspDetector()
    gd.main()
