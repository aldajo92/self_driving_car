#!/usr/bin/python

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()


def messageCallback(msg):

    global bridge
    img = bridge.imgmsg_to_cv2(msg)
    print('{0} : {1}'.format(msg.header.seq, img.shape))

    cv2.imshow('camera', img)
    cv2.waitKey(100)


def main():

    rospy.init_node('camera_viewer')

    subscriber = rospy.Subscriber(
        'self_driving_car/camera/image_color/BGR/raw',
        Image,
        queue_size=1,
        callback=messageCallback)

    cv2.namedWindow('camera')

    rospy.spin()


if __name__ == '__main__':
    main()
