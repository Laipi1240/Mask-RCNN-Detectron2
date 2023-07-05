import detectron2
from detectron2.utils.logger import setup_logger
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
from glob import glob

#import the COCO Evaluator to use the COCO Metrics
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wamv_train",)
cfg.DATASETS.TEST = ("wamv_val", )
cfg.DATALOADER.NUM_WORKERS = 0 #Single thread
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # datasets classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # datasets classes
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
cfg.MODEL.WEIGHTS = "online_model/model_final.pth"
#cfg.MODEL.WEIGHTS = "wamv_model/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
#register_coco_instances('wamv_test', {}, 
#                        '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/real_dataset.json', 
#                       '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/real_dataset')
register_coco_instances('wamv_test', {}, 
                        '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/val.json', 
                       '/home/arg/Mask-RCNN-Detectron2/datasets/split_dir/val')
cfg.DATASETS.TEST = ("wamv_test", )

# Create predictor
predictor = DefaultPredictor(cfg)
#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("wamv_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "wamv_test")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)

