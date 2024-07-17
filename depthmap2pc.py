import numpy as np
import laspy
import matplotlib.pyplot as plt
import open3d as o3d
import os
import math
#1.00672	0.00000	0.00000	0.00000
# 0.00000	1.73205	0.00000	0.00000
# 0.00000	0.00000	-1.00004	-0.20000
# 0.00000	0.00000	-1.00000	0.00000
#from depth map to point cloud data
matrix_path = '.\\pc\\638563122256124147_ProjectionMatrix.txt'
depth_path = "pc/638563122256124147_RenderResult.png"
def convert_to_pc(prefix):
    matrix_txt = np.loadtxt("pc/{}_ProjectionMatrix.txt".format(prefix))
    matrix = np.zeros((4,4))
    matrix = matrix_txt.tolist()
    matrix = np.array(matrix)
    #load depth map
    depth = plt.imread("pc/{}_RenderResult.png".format(prefix))
    depth = np.array(depth)
    #convert depth map to point cloud data
    height = depth.shape[0]
    width = depth.shape[1]
    point_cloud = []
    for i in range(height):
        for j in range(width):
            z = depth[i][j][0]
            # return 1.0 / (zBufferParam.z * depth + zBufferParam.w);
            far = 5000 
            near = 0.1
            para_z = near / (far -near)
            para_w = -near/(far -near)
            #z = -1.0 / (z * para_z + para_w)
            z = 1/z
            
            #gamma correction 
            #z = 1.0 / (z * 10.0 + 1.0)
            x = (j - width / 2) * z / matrix[0][0]/width
            y = (i - height / 2) * z / matrix[1][1]/height
            point_cloud.append([x, y, z])

    #visualize point cloud data
    point_cloud = np.array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    #save point cloud data
    o3d.io.write_point_cloud("pc/{}_point_cloud.ply".format(prefix), pcd)

def genscanpc(prefix):
    matrix_txt = np.loadtxt("pc/{}_ProjectionMatrix.txt".format(prefix))
    matrix = np.zeros((4,4))
    matrix = matrix_txt.tolist()
    matrix = np.array(matrix)
    #load depth map
    depth = plt.imread("pc/{}_RenderResult.png".format(prefix))
    depth = np.array(depth)
    #convert depth map to point cloud data
    height = depth.shape[0]
    width = depth.shape[1]
    point_cloud = []
    color_cloud = []
    red = np.asarray([1,0,0])
    green = np.asarray([0,1,0])
    p = 0
    size = 1000000
    for p in range(size):
            x_scale = math.cos(math.cos(p/3600*3.14))
            y_scale = math.sin(math.sin(p/3600*3.14))
            θ =p/100
            n=45
            x_dir = math.cos(n * θ) * math.cos(θ)
            y_dir = math.cos(n * θ) * math.sin(θ)
            i = int((x_dir+1)/2*height)
            j = int((y_dir+1)/2*width)
            i= min(i,height-1)
            i= max(i,0)
            j= min(j,width-1)
            j= max(j,0)
            z = depth[int(i)][int(j)][0]
            # return 1.0 / (zBufferParam.z * depth + zBufferParam.w);
            far = 5000 
            near = 0.1
            para_z = near / (far -near)
            para_w = -near/(far -near)
            #z = -1.0 / (z * para_z + para_w)
            z = 1/z
            
            #gamma correction 
            #z = 1.0 / (z * 10.0 + 1.0)
            x = (j - width / 2) * z / matrix[0][0]/width
            y = (i - height / 2) * z / matrix[1][1]/height
            point_cloud.append([x, y, z])
            color_cloud.append(red*1/size+(1-(p/size))*green)

    #visualize point cloud data
    point_cloud = np.array(point_cloud)
    color_cloud = np.array(color_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color_cloud)
    o3d.visualization.draw_geometries([pcd])
    #save point cloud data
    o3d.io.write_point_cloud("pc/{}_scan_point_cloud.ply".format(prefix), pcd)

for file in os.listdir("pc"):
    if file.endswith("ProjectionMatrix.txt"):
        prefix = file.split("_")[0]
        genscanpc(prefix)
        