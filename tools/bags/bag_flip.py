import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


input_bag = "rb5-bags/0510_0919/0510_0919_2023-05-10-09-19-25_0.bag"
topics = ['/camera_left/color/image_raw/compressed', '/camera_middle/color/image_raw/compressed', '/camera_right/color/image_raw/compressed']

output_bag = "output.bag"
flipped_topics = ['/camera_left/color/image_raw', '/camera_middle/color/image_raw', '/camera_right/color/image_raw']

bridge = CvBridge()

with rosbag.Bag(output_bag, 'w') as bag:
    with rosbag.Bag(input_bag, 'r') as input_bag:
        for topic, msg, t in input_bag.read_messages(topics=topics):
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            flipped_img = cv2.flip(img, 0)

            flipped_msg = bridge.cv2_to_imgmsg(flipped_img, encoding='passthrough')
            flipped_msg.header = msg.header

            topic_index = topics.index(topic)
            flipped_topic = flipped_topics[topic_index]
            bag.write(flipped_topic, flipped_msg, t)
#bag.close()
