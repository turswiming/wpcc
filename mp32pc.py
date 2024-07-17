#read mp3
from pydub import AudioSegment
import numpy as np
import open3d as o3d
import json
import os
path = "data_output/0001_save"
originpath = "data_input/"+path.split("/")[1].split("_")[0]+"_point_cloud.ply"

audio_x =  AudioSegment.from_file(path+"/x.ogg",format="ogg")
audio_y =  AudioSegment.from_file(path+"/y.ogg",format="ogg")
audio_z =  AudioSegment.from_file(path+"/z.ogg",format="ogg")

audio_x = audio_x.set_frame_rate(88200)
audio_y = audio_y.set_frame_rate(88200)
audio_z = audio_z.set_frame_rate(88200)

np_x = np.array(audio_x.get_array_of_samples())
np_y = np.array(audio_y.get_array_of_samples())
np_z = np.array(audio_z.get_array_of_samples())
# 对 np_x 进行插值
# x_new_indices = np.linspace(0, len(np_x) - 1, num=len(np_x) * 2)
# np_x = np.interp(x_new_indices, np.arange(len(np_x)), np_x)

# # 对 np_y 进行插值
# y_new_indices = np.linspace(0, len(np_y) - 1, num=len(np_y) * 2)
# np_y = np.interp(y_new_indices, np.arange(len(np_y)), np_y)

# # 对 np_z 进行插值
# z_new_indices = np.linspace(0, len(np_z) - 1, num=len(np_z) * 2)
# np_z = np.interp(z_new_indices, np.arange(len(np_z)), np_z)

print(np_x[:10])
#convert int16 to float64
np_x = np_x.astype(np.float64)
np_y = np_y.astype(np.float64)
np_z = np_z.astype(np.float64)

json_path = path+"/json.json"
with open(json_path, "r") as f:
    json_data = json.load(f)
x_min = json_data["x_min"]
x_max = json_data["x_max"]
y_min = json_data["y_min"]
y_max = json_data["y_max"]
z_min = json_data["z_min"]
z_max = json_data["z_max"]
x_starter = json_data["x_starter"]
y_starter = json_data["y_starter"]
z_starter = json_data["z_starter"]
#convert to point cloud

np_x = (np_x ) / (32767 *65534)
np_y = (np_y ) / (32767 *65534)
np_z = (np_z ) / (32767 *65534)
# np_x = pow(5, np_x)
# np_y = pow(5, np_y)
# np_z = pow(5, np_z)
np_x = np_x * max(abs(x_min), abs(x_max))
np_y = np_y * max(abs(y_min), abs(y_max))
np_z = np_z * max(abs(z_min), abs(z_max))


point_cloud = np.stack((np_x, np_y, np_z), axis=-1)
print(point_cloud.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
#add colors
size = point_cloud.shape[0]

originpcd = o3d.io.read_point_cloud(originpath)
#color origin point cloud
np_colors_origin = np.zeros((len(originpcd.points), 3))
for i in range(len(originpcd.points)):
    np_colors_origin[i] = [0,0, 1]

np_colors = np.zeros((size, 3))
originpcd_points = np.asarray(originpcd.points)

diff = np.abs(point_cloud-originpcd_points[:size])
diff = diff/np.max(diff)
for i in range(size):
    np_colors[i] = diff[i]
pcd.colors = o3d.utility.Vector3dVector(np_colors)
originpcd.colors = o3d.utility.Vector3dVector(np_colors_origin)
o3d.io.write_point_cloud(path+"/point_cloud.ply", pcd)
#visualize point cloud
o3d.visualization.draw_geometries([pcd])