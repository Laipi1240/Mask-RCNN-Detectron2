import os
import cv2
from cv_bridge import CvBridge
import rosbag

class arg_bag_extracter:
    def __init__(self, bag_file, output_dir, image_topic): 
        self.bag_file = bag_file
        self.bag = rosbag.Bag(self.bag_file, "r")
        self.output_dir = output_dir
        self.image_topic = image_topic

    def extract(self):
        bridge = CvBridge()
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=[self.image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(os.path.join(self.output_dir, "color%06i.jpg" % count), cv_img)
            count += 1
        self.bag.close()
        print("done, count is "+ str(count) + " and output to " + self.output_dir)

#path = "/home/arg/Mask-RCNN-Detectron2/tools/bags/rb5-bags/"
path = "/home/chenyi/2T/Mask-RCNN-Detectron2/tools/bags/rb5-bags/"
#image_topic = "/d435_backward/color/image_raw"
image_topic = "/camera_middle/color/image_raw/compressed"
bag_name = path + "0511_drone_compressed"
print(bag_name)
bag_file = bag_name + ".bag"
print(bag_file)
output_dir = bag_name + "/"
print(output_dir)
os.mkdir(output_dir)
extracter = arg_bag_extracter(bag_file, output_dir, image_topic)
extracter.extract()

