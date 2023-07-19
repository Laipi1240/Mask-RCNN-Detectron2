#!/usr/bin/env python3

import detectron2
from detectron2.utils.logger import setup_logger
import rospy
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import gdown
import torch
import math

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wamv_train",)
cfg.DATASETS.TEST = ("wamv_val", )
cfg.DATALOADER.NUM_WORKERS = 0 #Single thread
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # datasets classes
cfg.SOLVER.IMS_PER_BATCH = 4 #Batch size
ITERS_IN_ONE_EPOCH = 610 #dataset_imgs/batch_size
cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 50) # ITERS
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.GAMMA = 0.1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH #Save training model interval
cfg.MODEL.WEIGHTS = "/home/arg/Mask-RCNN-Detectron2/tools/wamv_only/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
register_coco_instances('wamv_val', {}, 
                        '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/val.json', 
                       '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/val')
cfg.DATASETS.TEST = ("wamv_val", )
subt_metadata = MetadataCatalog.get("wamv_val")
dataset_dicts = DatasetCatalog.get("wamv_val")


class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.detection_image = None
        self.br = CvBridge()
        self.predictor = DefaultPredictor(cfg)
        self.rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher('/detectron2/detections', Image,queue_size=10)
        # Subscribers
        rospy.Subscriber("/detectron2/image_raw",Image,self.callback)

    def callback(self, msg):
        rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg)
        self.image = cv2.flip(self.image, 0)
        outputs = self.predictor(self.image)
        v = Visualizer(self.image[:, :, ::-1], metadata=subt_metadata, scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        self.detection_image = v.get_image()[:, :, ::-1]
        rospy.loginfo('Prediction Done')

    def start(self):
        rospy.loginfo("Timing images")
        #rospy.spin()
        while not rospy.is_shutdown():
            #br = CvBridge()
            if self.detection_image is not None:
                self.pub.publish(self.br.cv2_to_imgmsg(self.detection_image))
            self.rate.sleep()
            
if __name__ == '__main__':
    rospy.init_node("detectron2_maskrcnn", anonymous=True)
    my_node = Nodo()
    my_node.start()
