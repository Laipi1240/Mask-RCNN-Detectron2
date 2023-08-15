import os
import cv2
from cv_bridge import CvBridge
import rosbag
import numpy as np

class arg_bag_extracter:
    def __init__(self, bag_file, output_dir, image_topic): 
        self.bag_file = bag_file
        self.bag = rosbag.Bag(self.bag_file, "r")
        self.output_dir = output_dir
        self.image_topic = image_topic

    def extract(self):
        bridge = CvBridge()
        count = 0
        dismiss = 0
        for topic, msg, t in self.bag.read_messages(topics=[self.image_topic]):
            if dismiss < 8:
                dismiss += 1
                continue
            dismiss = 0
            count += 1
            np_arr = np.frombuffer(msg.data, np.uint8)
            # Decode the compressed image
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_img = cv2.flip(cv_img, 0)
            #cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(os.path.join(self.output_dir, "color%06i.jpg" % count), cv_img)
        self.bag.close()
        print("done, count is "+ str(count) + " and output to " + self.output_dir)

path = "/home/arg/Mask-RCNN-Detectron2/tools/bags/0721/"
#path = "/home/chenyi/2T/Mask-RCNN-Detectron2/tools/bags/rb5-bags/"
#image_topic = "/d435_backward/color/image_raw"
image_topic = "/camera_left/color/image_raw/compressed"
bags = os.listdir(path)
for bag in bags:
    bag_name = path + bag
    bag_file = bag_name
    print(bag_file)
    output_dir = path + bag[:-4] + "_left/"
    print(output_dir)
    os.mkdir(output_dir)
    extracter = arg_bag_extracter(bag_file, output_dir, image_topic)
    extracter.extract()
