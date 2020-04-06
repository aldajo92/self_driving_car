# Self-Driving Car over rc car 1/10 scale

# Introduction
This project pretend to modify an existent rc car, replacing and adding some parts to crate a complete platform to develop and test algoritms related with self-driving car.

# Jetson Nano configuration
Nvidia has a complete support to install the OS in the jetson nano. In the [Prepare for Setup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#prepare) section, of the jetson nano support page, you can found instructions to prepare the microSD and the power supply specs required for the install process. 

# Install Process

## Install ROS:
The ROS version used for this project is ROS-Melodic, wich is supported for jetson nano, the entire install process requires to follow this tutorial:

https://www.stereolabs.com/blog/ros-and-nvidia-jetson-nano/


## Install i2c librarie for jetson nano:
Based on this [tutorial](https://www.jetsonhacks.com/2019/07/22/jetson-nano-using-i2c/) this are the commands required:

```
$ git clone https://github.com/JetsonHacksNano/ServoKit
$ cd ServoKit
$ ./installServoKit.sh
```

To check if the i2c is discovered in the OS, use this command:
```
$ i2cdetect -y -r 1
```

## Install ros streamming:
```
$ sudo apt install -y \
    tmux \
    ros-melodic-image-view \
    ros-melodic-web-video-server \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio
```

## install jupyter-lab

```
$ pip install jupyterlab
```

## References ##
- [jetson nano - i2c](https://www.jetsonhacks.com/2019/07/22/jetson-nano-using-i2c/)
- [roskeycar os preparation](https://github.com/roboticamed/ROSkey-car/blob/master/doc/OS-preparation.md)
- [IPython Kernel](https://ipython.readthedocs.io/en/latest/install/kernel_install.html)
- [ROS Creating a ROS Package](http://wiki.ros.org/ROS/Tutorials/CreatingPackage)
- [ROS Writing a Simple Publisher and Subscriber (Python)](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29)
- [ROS web_video_server](https://wiki.ros.org/web_video_server)
- [Converting between ROS images and OpenCV images (Python)](http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython)