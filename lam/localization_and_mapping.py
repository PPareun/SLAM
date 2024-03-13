import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import cv_bridge
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import os
from plyfile import PlyData, PlyElement
import time
class ICP():
    def __init__(self,init_R=[[1,0,0],
        [0,1,0],
        [0,0,1]],init_t=[0,0,0]):
        self.R = np.array(init_R)
        self.t = np.array(init_t)
        self.r_max = 17
        self.d_min = .7
        self.s_sum = 0
        self.count = 0
        self.thres = 1.2
        self.flag = 0


    def find_correspondence(self,source,target):
        tree = cKDTree(target)
        distances, indices = tree.query(source, k = 1)
        mask = np.array(distances) < self.thres
        correspondence_points = [np.arange(len(source))[mask], indices[mask]]
        rev_mask = np.array(distances) > self.thres
        return correspondence_points, np.arange(len(source))[rev_mask]

    def get_transformation_svd(self, source_point, correspondence_point):
        # source_point: Nx3 The source point
        # correspondence_poin:t Nx3, The correspondence points of source_point.
        # For example, A correspondence point of source_point[i] is correspondence_point[i]
        # Step 1: Align center point
        source_centroid = self.compute_centroid(source_point)
        target_centroid = self.compute_centroid(correspondence_point)

        source_centered = source_point - source_centroid
        target_centered = correspondence_point - target_centroid

        # Step 2: Compute the Cross-Covariance Matrix
        H = source_centered.T @ target_centered
        # Step 3: Apply SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Step 4: Ensure a proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # Step 5: Determine the Optimal Rotation and Translation
        t = target_centroid - R @ source_centroid
        return R ,t

    def compute_centroid(self,points):
        return np.mean(points, axis=0)

    def icp_svd(self,current_pt,target_pt,old_r,old_t):
        # [1-1] Find correspondence
        old_M = np.concatenate((np.concatenate((old_r, np.expand_dims(old_t, 0).T), axis = -1), np.array([[0,0,0,1]])), axis = 0)
        current_pt = np.concatenate((np.array(current_pt), np.ones(len(current_pt)).reshape(-1,1)), axis = -1)
        transformed_source_point = (old_M @ current_pt.T).T[:,:3]
        indices, sc_indices = self.find_correspondence(transformed_source_point,target_pt)
        conrrespondence_point = target_pt[:, :3]
        sc_indices = transformed_source_point[sc_indices]
        R, t = self.get_transformation_svd(transformed_source_point[indices[0]][:, [0,2]], conrrespondence_point[indices[1]][:, [0,2]])
        delta_r = 2 * self.r_max * np.sin(0.5 * np.arccos((np.trace(R) - 1) / 2))
        delta_t = np.linalg.norm(t)
        delta = delta_r + delta_t
        if delta > self.d_min:
            self.s_sum += delta ** 2
            self.count += 1
            self.thres = min(np.sqrt(self.s_sum / self.count), self.thres)
        R_3D = np.eye(3)
        R_3D[0][0], R_3D[0][2], R_3D[2][0], R_3D[2][2] = R[0][0], R[0][1], R[1][0], R[1][1]
        t_3D = np.zeros(3)
        t_3D[0], t_3D[2] = t[0], t[1]
        new_M = np.concatenate((np.concatenate((R_3D, np.expand_dims(t_3D, 0).T), axis = -1), np.array([[0,0,0,1]])), axis = 0) @ old_M

        transformed_source_points = new_M @ current_pt.T

        R, t = new_M[:3,:3], new_M[:3, -1]

        return transformed_source_points.T[:, :3],indices,R,t, sc_indices
    
    def error_evaluation(self, transformed_source_points, target_points, indices):
        # Write down your implementation
        loss = np.mean((transformed_source_points[indices[0]] - target_points[indices[1]]) ** 2)
        return loss

# Localization and Mapping node
class LAM(Node):
    def __init__(self):
        super().__init__('icp_depth_subscriber')
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/point_cloud', 10)
        self.odom_publisher = self.create_publisher(Odometry, '/carter_odom', 10)
        self.subscription = self.create_subscription(Image,
            '/carter1/depth_left',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        self.map_points = []
        self.view_points = []
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.t = np.array([0,0,0])
        self.icp = ICP()    
        self.count = 0
        self.cc = 0
        self.map_save = False

    def voxel_downsample(self, points, voxel_size = .07):
        points_np = np.array(points)
        assert points_np.shape[1] == 3
        voxel_grid = {}
        for p in points_np:
            voxel = tuple(np.floor(p / voxel_size).astype(int))
            voxel_grid[voxel] = p
        downsampled_points = list(voxel_grid.values())

        return np.array(downsampled_points)

    def voxel_downsample_2d(self, points, voxel_size=0.1):
        points_np = np.array(points)
        assert points_np.shape[1] == 3
        voxel_grid = {}

        for p in points_np:
            p_2d = np.array([p[0], 0, p[2]])
            voxel = tuple(np.floor(p_2d / voxel_size).astype(int))
            voxel_grid[voxel] = p_2d

        downsampled_points = list(voxel_grid.values())
        
        return np.array(downsampled_points)
        
    def saveply(self, filename='map.ply'):
        save_points = self.view_points if self.map_save else self.map_points
        vertex = np.array([(p[0], p[1], p[2]) for p in save_points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)

    def save_odom(self, rotation_matrix, translation, timestamp, timestamp_nsec):
        position = translation
        orientation = R.from_matrix(rotation_matrix).as_quat()

        odom_data = f"{timestamp}.{timestamp_nsec} {position[0]} {position[1]} {-position[2]} {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}\n"

        file_path = "odometry_data.txt"

        with open(file_path, "a" if os.path.exists(file_path) else "w") as file:
            file.write(odom_data)

    def rgb_callback(self,msg):
        pass

    def compute_icp(self,source_points,target_points,init_rotation,init_translation,max_iterations=300):
        convergence_threshold = 0.001
        rot, trans = init_rotation, init_translation
        for iteration in range(max_iterations):
            transformed_source_points,indices,rot,trans, sc_indices = self.icp.icp_svd(source_points,target_points,rot,trans)
            loss = self.icp.error_evaluation(transformed_source_points, target_points, indices)        
            if loss < convergence_threshold:
                break
        return transformed_source_points,rot,trans, sc_indices
    
    def depth_callback(self, depth_msg):
        self.start_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))
            return

        pt = self.convert_to_point_cloud(cv_image)
        if self.map_save:
            view_pt = pt[(pt[:,1] > -.5) & (pt[:,1] < .2)]
        pt = pt[(pt[:,1] > -.5) & (pt[:,1] < -.2)]
        pt = self.voxel_downsample_2d(pt)

        if len(self.map_points) != 0:
            # Compute ICP
            source_points = pt
            target_points = np.array(self.map_points)
            transformed_source_points, rot, trans, sc_indices = self.compute_icp(source_points, target_points, self.R, self.t)
            if self.map_save:
                old_M = np.concatenate((np.concatenate((rot, np.expand_dims(trans, 0).T), axis = -1), np.array([[0,0,0,1]])), axis = 0)
                view_pt = np.concatenate((np.array(view_pt), np.ones(len(view_pt)).reshape(-1,1)), axis = -1)
                self.view_points.extend((old_M @ view_pt.T).T[:,:3].tolist())
                self.view_points = self.voxel_downsample(self.view_points).tolist()
            # Add the point P_t to map
            self.map_points.extend(sc_indices)
            self.R, self.t = rot, trans
        else:
            self.map_points.extend(pt)
        
        
        self.map_points = self.voxel_downsample_2d(self.map_points).tolist()
        if self.map_save:
            self.saveply()
        save_points = self.view_points if self.map_save else self.map_points
        self.publish_pointcloud(np.array(save_points))
        # publish odometry
        self.publish_odometry(self.R, self.t, depth_msg.header.stamp.sec, depth_msg.header.stamp.nanosec)



    def publish_pointcloud(self, np_point, frame_id='map'):
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = np_point.astype(dtype).tobytes()

        fields = [PointField(name = n, offset = i * itemsize, datatype = ros_dtype, count = 1) for i, n in enumerate('xyz')]

        header = Header(frame_id = frame_id)

        pc2_msg = PointCloud2(header = header, height = 1, width = np_point.shape[0], is_dense = False, is_bigendian = False, fields = fields, point_step = (itemsize * 3), row_step = (itemsize * 3* np_point.shape[0]), data = data)

        self.point_cloud_publisher.publish(pc2_msg)

    def convert_to_point_cloud(self, cv_image):

        z = cv_image.reshape(-1 ,1)

        fx, fy, cx, cy = 305.41638, 407.85113, 320.0, 240.0 
        y, x = np.indices((cv_image.shape[0], cv_image.shape[1]))
        x, y = (x - cx) * cv_image / fx, (y - cy) * cv_image / fy

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        valid_mask = np.logical_and(z > 0, np.isfinite(z))
        point_cloud = np.stack((x, y, z), axis=-1)
        point_cloud_np = point_cloud[valid_mask]
        
       
        return point_cloud_np
    
    def publish_odometry(self,rotation_matrix, translation, timestamp, timestamp_nsec):
        end_time = time.time()
        time_taken= end_time - self.start_time
        print(time_taken)
        frame_id='map'
        odom_msg = Odometry()

        odom_msg.header.frame_id = frame_id
        odom_msg.pose.pose.position.x = float(translation[0])
        odom_msg.pose.pose.position.y = float(translation[1])
        odom_msg.pose.pose.position.z = float(translation[2])

        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()

        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]

        odom_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_publisher.publish(odom_msg)
        self.save_odom(rotation_matrix, translation, timestamp, timestamp_nsec)


def main(args=None):
    rclpy.init(args=args)
    icp_depth_subscriber = LAM()
    rclpy.spin(icp_depth_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
