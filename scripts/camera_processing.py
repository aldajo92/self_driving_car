#!/usr/bin/python

import cv2
import numpy as np
import rospy
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    left_side_x = []
    left_side_y = []
    right_side_x = []
    right_side_y = []
    slp = 0.4

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > slp:
                left_side_x.extend((x1,x2))
                left_side_y.extend((y1,y2))
            if slope < -slp: 
                right_side_x.extend((x1,x2))
                right_side_y.extend((y1,y2))
         
    height = []
    height.extend((img.shape[0],min(left_side_y+right_side_y)))
    
    left_slp = np.polyfit(left_side_y,left_side_x,1)
    right_slp = np.polyfit(right_side_y,right_side_x,1)
    
    leftV = np.poly1d(left_slp)
    rightV = np.poly1d(right_slp)
    
    left_x1 = int(leftV(height[0]))
    left_x2 = int(leftV(height[1]))
    right_x1 = int(rightV(height[0]))
    right_x2 = int(rightV(height[1]))

    cv2.line(img,(left_x1,height[0]),(left_x2,height[1]),color,thickness)
    cv2.line(img,(right_x1,height[0]),(right_x2,height[1]),color,thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, betha=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, betha, gamma)

def process_lines(image):
    (height, width) = image.shape[:2]
    ## Convert to grayscale
    img_gray = np.copy(image)
    img_gray = grayscale(img_gray)

    ## Apply Gaussian smooth
    img_smooth = gaussian_blur(img_gray, 7)

    ## Apply Canny
    low_threshold = 80
    high_threshold = 150
    edges = canny(img_smooth, low_threshold, high_threshold)
    
    top_reduction = 25

    ## Region of interest
    left_top = [0, height-1-top_reduction]
    right_top = [width-1, height-1-top_reduction]
    
    left_bottom = [0, height-1]
    right_bottom = [width-1, height-1]
    apex = [width//2, top_reduction]
    area = np.array( [[left_top, left_bottom, right_bottom, right_top, apex]], dtype=np.int32 )
    img_region = region_of_interest(edges, area)

    # Get Lines
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 15 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    img_lines = hough_lines(img_region, rho, theta, threshold, min_line_len, max_line_gap)

    ## Draw Lines
    result = weighted_img(img_lines, image)
    
    return result


# image = cv2.imread('test_images/snapshot.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class CameraProcessor(object):

    def __init__(self):

        self.imgSubscriber = rospy.Subscriber(
            'self_driving_car/camera/image_color/BGR/raw',
            Image,
            queue_size=1,
            callback=self.onImageReceived)

        self.imgBinaryPublisher = rospy.Publisher(
            'self_driving_car/camera/lines/raw', Image, queue_size=1)

        self.donkeyPublisher = rospy.Publisher(
            '/motor/twist', Twist, queue_size=1)

        self.bridge = CvBridge()

        # HSV color threshold for blue markers
        self.colorMin = np.array([75, 130, 60], dtype=np.uint8)
        self.colorMax = np.array([130, 255, 255], dtype=np.uint8)

        # structuring element for morphology operations
        self.morphKernel = np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]], dtype=np.uint8)

        # input image
        self.BGR = None

    def onImageReceived(self, msg):
        """
        Callback for receiving image messages

        Parameters
        ----------
        msg : sensor_msgs.msg.Image.
            Image message
        """

        self.BGR = self.bridge.imgmsg_to_cv2(msg)
        self.processImage(self.BGR)

    def processImage(self, BGR):

        # reduce the resolution of the image to half to allow for
        # faster processing
        BGR = cv2.resize(BGR, (320, 240))

        # convert image to HSV
#         HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
        try:
            result = process_lines(BGR)
        except:
            result = BGR
        self.imgBinaryPublisher.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))


def main():

    rospy.init_node('camera_processing')

    processor = CameraProcessor()

    rospy.spin()


if __name__ == '__main__':
    main()
