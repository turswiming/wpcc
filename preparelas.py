import laspy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

#read using laspy
las = laspy.read(".\data_input\las.las")
#get point cloud data
point_cloud = las.points
print(point_cloud)
gps_time = las.points['gps_time']
print(gps_time.shape)
#convert to numpy array
xyz = np.vstack((las.x, las.y, las.z)).transpose()
pcd_o3d = o3d.geometry.PointCloud()
print(xyz.shape)
xyz_np = np.zeros(xyz.shape)
for i in range(xyz.shape[0]):
    xyz_np[i] = [xyz[i][0], xyz[i][1], xyz[i][2]]

pcd_o3d.points = o3d.utility.Vector3dVector(xyz_np)
#save point cloud data
o3d.io.write_point_cloud("data_input/0001_point_cloud.ply", pcd_o3d)