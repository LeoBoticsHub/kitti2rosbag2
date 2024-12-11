#!/usr/bin/env python3

import rclpy
import numpy as np
import cv2
import rosbag2_py
import os
import shutil
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from kitti2rosbag2.utils.kitti_utils import KITTIOdometryDataset
from kitti2rosbag2.utils.quaternion import Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.serialization import serialize_message
from tf2_msgs.msg import TFMessage

class Kitti_Odom(Node):
    def __init__(self):
        super().__init__("kitti_rec")

        # Declare ROS parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('sequence', rclpy.Parameter.Type.INTEGER),
                ('gray_dir', rclpy.Parameter.Type.STRING),
                ('velodyne_dir', rclpy.Parameter.Type.STRING),
                ('odom', rclpy.Parameter.Type.BOOL),
                ('odom_dir', rclpy.Parameter.Type.STRING),
                ('bag_dir', rclpy.Parameter.Type.STRING),
            ]
        )

        # Get ROS parameters
        sequence     = self.get_parameter('sequence').value
        gray_dir     = self.get_parameter('gray_dir').get_parameter_value().string_value
        velodyne_dir = self.get_parameter('velodyne_dir').get_parameter_value().string_value
        odom         = self.get_parameter('odom').value
        bag_dir      = self.get_parameter('bag_dir').get_parameter_value().string_value

        if odom == True:
            odom_dir = self.get_parameter('odom_dir').get_parameter_value().string_value
        else:
            odom_dir = None
        
        # Create and handle the Kitti dataset
        self.kitti_dataset = KITTIOdometryDataset(gray_dir, velodyne_dir, sequence, odom_dir) # init the Kitti dataset
        self.bridge = CvBridge() # init the CV bridge for image conversions
        self.counter = 0 # init the frame counter
        self.counter_limit = len(self.kitti_dataset.left_images()) - 1  # the total number of frames in the sequence (it is the same for cameras, velodyne ...)
        
        # Get data
        self.left_imgs   = self.kitti_dataset.left_images()  # left camera images
        self.right_imgs  = self.kitti_dataset.right_images() # right camera images
        self.point_cloud = self.kitti_dataset.point_cloud()  # point cloud
        self.times_file = self.kitti_dataset.times_file()    # time
        self.odom = odom # TODO: remove and use just odom
        if odom == True:
            try:
                self.ground_truth = self.kitti_dataset.odom_pose() # ground truth poses
            except FileNotFoundError as filenotfounderror:
                self.get_logger().error("Error: {}".format(filenotfounderror))
                rclpy.shutdown()
                return

        # Init rosbag writer
        self.writer = rosbag2_py.SequentialWriter()
        # Create the bag file only if the specified directory does not alreadty exist
        if os.path.exists(bag_dir):
            shutil.rmtree(bag_dir, ignore_errors=True)
        
        storage_options   = rosbag2_py._storage.StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        # Define topics and their data for rosbag
        left_img_topic_info    = rosbag2_py._storage.TopicMetadata(name='/camera2/left/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr')
        right_img_topic_info   = rosbag2_py._storage.TopicMetadata(name='/camera3/right/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr')
        point_cloud_topic_info = rosbag2_py._storage.TopicMetadata(name='/car/point_cloud', type='sensor_msgs/msg/PointCloud2', serialization_format='cdr')
        odom_topic_info        = rosbag2_py._storage.TopicMetadata(name='/car/base/odom', type='nav_msgs/msg/Odometry', serialization_format='cdr')
        tf_topic_info          = rosbag2_py._storage.TopicMetadata(name='/tf', type='tf2_msgs/msg/TFMessage', serialization_format='cdr')  # Fixed type for /tf topic
        path_topic_info        = rosbag2_py._storage.TopicMetadata(name='/car/base/odom_path', type='nav_msgs/msg/Path', serialization_format='cdr')
        left_cam_topic_info    = rosbag2_py._storage.TopicMetadata(name='/camera2/left/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr')
        right_cam_topic_info   = rosbag2_py._storage.TopicMetadata(name='/camera3/right/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr')

        # Create the topics in the rosbag file
        self.writer.create_topic(left_img_topic_info)
        self.writer.create_topic(right_img_topic_info)
        self.writer.create_topic(point_cloud_topic_info)
        self.writer.create_topic(odom_topic_info)
        self.writer.create_topic(tf_topic_info)
        self.writer.create_topic(path_topic_info)
        self.writer.create_topic(left_cam_topic_info)
        self.writer.create_topic(right_cam_topic_info)

        # Initialize Path message for recording odometry paths
        self.p_msg = Path()

        # Start the recording timer
        self.timer = self.create_timer(0.05, self.rec_callback)


    def rec_callback(self):
        ''' Callback for recording data to rosbag 
        '''
        time = self.times_file[self.counter]
        timestamp_ns = int(time * 1e9) # Convert time to nanoseconds

        # retrieving images and writing to bag
        left_image    = cv2.imread(self.left_imgs[self.counter])
        right_image   = cv2.imread(self.right_imgs[self.counter])
        left_img_msg  = self.bridge.cv2_to_imgmsg(left_image, encoding='passthrough')
        right_img_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='passthrough')
        self.writer.write('/camera2/left/image_raw', serialize_message(left_img_msg), timestamp_ns)
        self.writer.write('/camera3/right/image_raw', serialize_message(right_img_msg), timestamp_ns)

        # retrieving project matrix and writing to bag
        p_mtx2 = self.kitti_dataset.projection_matrix(1)
        self.rec_camera_info(p_mtx2, '/camera2/left/camera_info', timestamp_ns)
        p_mtx3 = self.kitti_dataset.projection_matrix(2)
        self.rec_camera_info(p_mtx3, '/camera3/right/camera_info', timestamp_ns)

        # Retrieve and write point cloud data to the bag
        point_cloud_data = self.point_cloud[self.counter]  # Get the current frame's point cloud
        #self.rec_point_cloud(point_cloud_data, timestamp_ns)

        # Record odometry data if enabled
        if self.odom == True:
            translation = self.ground_truth[self.counter][:3,3]
            rot =  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) * self.ground_truth[self.counter][:3, :3] # [[1, 0, 0], [0, 0, 1], [0, -1, 0]]) 
            quaternion = Quaternion()
            # quaternion = quaternion.rotationmtx_to_quaternion(rot)
            quaternion = quaternion.rotation_matrix_to_quaternion(rot)           
            quaternion = quaternion / np.linalg.norm(quaternion)
            self.rec_odom_msg(translation, quaternion, timestamp_ns)
            self.rec_odom_tf(translation, quaternion, timestamp_ns)


        self.get_logger().info(f'{self.counter}-Images Processed')

        # Stop recording if all frames are processed  
        if self.counter >= self.counter_limit:
            self.get_logger().info('All images and poses published. Stopping...')
            rclpy.shutdown()
            self.timer.cancel()

        self.counter += 1
        return
    
    def rec_point_cloud(self, point_cloud_data, timestamp_ns):
        """Converts point cloud data to a ROS2 PointCloud2 message and writes it to the bag.

        Args:
            point_cloud_data (numpy.ndarray): NumPy array of shape (N, 4) with [x, y, z, intensity].
            timestamp_ns (int): Timestamp in nanoseconds.
        """
        from sensor_msgs.msg import PointCloud2, PointField
        import struct

        # Initialize PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = "odom"  # Frame of the Velodyne data
        cloud_msg.header.stamp.sec = timestamp_ns // int(1e9)  # Seconds part of timestamp
        cloud_msg.header.stamp.nanosec = timestamp_ns % int(1e9)  # Nanoseconds part of timestamp

        # Set PointCloud2 metadata
        cloud_msg.height = 1  # Unordered point cloud (1 row)
        cloud_msg.width = point_cloud_data.shape[0]  # Number of points
        cloud_msg.is_dense = True  # No invalid points (e.g., NaNs)
        cloud_msg.is_bigendian = False  # Assume little-endian format
        cloud_msg.point_step = 16  # Size of a point (x, y, z, intensity each 4 bytes)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width  # Size of a row in bytes

        # Define fields for x, y, z, and intensity
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Convert NumPy array to byte array
        points = []
        for point in point_cloud_data:
            x, y, z, intensity = point
            points.append(struct.pack('ffff', x, y, z, intensity))  # Pack into binary format
        cloud_msg.data = b''.join(points)  # Combine into a single byte array

        # Write to rosbag
        self.writer.write('/car/point_cloud', serialize_message(cloud_msg), timestamp_ns)


    def rec_camera_info(self, mtx, topic, timestamp):
        camera_info_msg_2 = CameraInfo()
        camera_info_msg_2.p = mtx.flatten()
        self.writer.write(topic, serialize_message(camera_info_msg_2), timestamp)   
        return
    
    def rec_odom_msg(self, translation, quaternion, timestamp):
        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "odom"

        odom_msg.pose.pose.position.x = translation[2]
        odom_msg.pose.pose.position.y = -translation[0]  
        odom_msg.pose.pose.position.z = -translation[1]
        
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]

        self.writer.write('/car/base/odom', serialize_message(odom_msg), timestamp)
        self.rec_path_msg(odom_msg, timestamp)
        return
    

    def rec_odom_tf(self, translation, quaternion, timestamp):
        """Publishes odometry data as a TransformStamped message to the /tf topic.

        Args:
            translation (numpy.ndarray): Translation vector (x, y, z).
            quaternion (numpy.ndarray): Quaternion (x, y, z, w).
            timestamp (int): Timestamp in nanoseconds.
        """
        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.header.stamp.sec = timestamp // int(1e9)
        transform.header.stamp.nanosec = timestamp % int(1e9)

        # Set translation and rotation
        transform.transform.translation.x = translation[2]
        transform.transform.translation.y = -translation[0]
        transform.transform.translation.z = -translation[1]

        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        
        # Create a TFMessage and add the TransformStamped
        tf_message = TFMessage()
        tf_message.transforms.append(transform)

        # Write the TFMessage to the /tf topic
        self.writer.write('/tf', serialize_message(tf_message), timestamp)
    
    def rec_path_msg(self, odom_msg, timestamp):
        pose= PoseStamped()
        pose.pose = odom_msg.pose.pose
        pose.header.frame_id = "odom"
        self.p_msg.poses.append(pose)
        self.p_msg.header.frame_id = "map"
        self.writer.write('/car/base/odom_path', serialize_message(self.p_msg), timestamp)
        return

def main(args=None):
    rclpy.init(args=args)
    node = Kitti_Odom()
    rclpy.spin(node)
    try:
        rclpy.shutdown()
    except Exception as e:
        pass

if __name__ == '__main__':
    main()