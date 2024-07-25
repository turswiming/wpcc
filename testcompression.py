"""
writen by: Ziqi Li

This file is used to compress point cloud data using fft and image compression method.

The main function is pc2mp3, which takes a path of a point cloud file as input, and output the compression ratio and PSNR of the compression.

usage:
    pcc = PCcompression(box_size, frame_size, compression_value, visualize, use8bit)
    pcc.pc2mp3(path)
    and the compression ratio and PSNR will be printed.
    
"""

import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
import struct
import open3d as o3d
import os
import json
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt
import math
import getScaleParameter as gsp
from sklearn.neighbors import NearestNeighbors
import numpy as np

x_real_original_global = np.array([])
class PCcompression:
    def __init__(self,frame_size,compression_value,ununiformlevel =3,visualize = False, use8bit =False) -> None:
        if frame_size % 2 != 0:
            raise ValueError("frame_size should be even")
        self.frame_size = frame_size
        self.compression_value = compression_value
        # if edge_size *2 >=frame_size:
        #     raise ValueError("edge_size should be smaller than frame_size/2")
        # self.edge_size = edge_size
        self.ununiformlevel = ununiformlevel
        self.visualize = visualize
        self.use8bit = use8bit
        self.tiny = 10000
        self.threshold = 0.005
        pass

        

    def dft(self,x):
        """
        Compute the Discrete Fourier Transform (DFT) of an array.
        
        Parameters:
        x (np.array): Input array.
        
        Returns:
        np.array: DFT of the input array.
        """
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        return np.dot(e, x)


    def savefft(self,path,fft_frames):
        #save fft frames
        with open(path, "wb") as f:
            for frame in fft_frames:
                f.write(struct.pack('f'*len(frame), *frame))

    def get_min_max(self,image, width, height):
        min = np.zeros((math.ceil(image.shape[0] / width), math.ceil(image.shape[1] / height)))
        max = np.zeros((math.ceil(image.shape[0] / width), math.ceil(image.shape[1] / height)))
        for i in range(0, image.shape[0], width):
            for j in range(0, image.shape[1], height):
                min[int(i / width), int(j / height)] = np.min(image[i:i + width, j:j + height])
                max[int(i / width), int(j / height)] = np.max(image[i:i + width, j:j + height])
        return min, max

    def sintransform(self,image, level):
        if level == 0:
            return image
        else:
            image = np.sin(image*3.1415926/2)
            return self.sintransform(image, level-1)
            
    def arcsintransform(self,image, level):
        if level == 0:
            return image
        else:
            image = np.arcsin(image)/(3.1415926/2)
            return self.arcsintransform(image, level-1)
    
    def arctantransform(self,image, level):
        if level == 0:
            return image
        else:
            image = np.arctan(image/(3.1415926/2))
            return self.arctantransform(image, level-1)
    def tantransform(self,image, level):
        if level == 0:
            return image
        else:
            image = np.tan(image)* 3.1415926 / 2
            return self.tantransform(image, level-1)
    
    def UnuniQuantize(self, image:np.array,level:float):
        return self.arcsintransform(image, level)
        return self.arctantransform(image, level)
        return np.arctan(image*level)*2/3.1415926
        return self.sintransform(image, level)
        pass
        
    def unpackUnuniQuantize(self, image:np.array,level:float):
        return self.sintransform(image, level)
        return self.tantransform(image, level)
        return np.tan((image)*3.1415926/2)/level
        return self.arcsintransform(image, level)
        pass
    
    def genlowPrecisionPic(self,image:np.array, threshold:float):
        
        mask = (image > -threshold) & (image < threshold)
        print("mask: ", mask[mask == True].shape)
        print("mask: ", mask[mask == False].shape)
        image_clamped = np.clip(image, -threshold, threshold)
        image_scaled = image_clamped /threshold
        
        mask_line = mask.reshape(-1)
        false_indices = np.where(mask_line == False)[0]
        return image_scaled, false_indices

    def rebuildHighPrecisionPic(self,lowPrecisionPic:np.array,highPrecisionNumbers:np.array,mask_line:np.array,threshold:float):
        image = lowPrecisionPic * threshold 
        for index in range(mask_line.shape[0]):
            indices = mask_line[index]
            x = indices//image.shape[1]
            y = indices%image.shape[1]
            image[x,y] = highPrecisionNumbers[index]
        return image
    
    def saveonechannel(self,value, prefix,channel_name) ->np.array:
        #1.1 cliping
        x_frames = []
        for i in range(0, len(value), self.frame_size):
            x_frames.append(value[i:i+self.frame_size])
        #1.2 DCT
        #apply DCT to each frame
        x_dct_frames = []
        max_length = 0
        for frame in x_frames:
            if len(frame) < self.frame_size:
                continue
            dct_result = dct(frame, norm='ortho')
            x_dct_frames.append(dct_result)
            if len(dct_result) > max_length:
                max_length = len(dct_result)
        
        x_dct_frames_array = np.array(x_dct_frames)
        real = np.zeros(x_dct_frames_array.shape)
        for i in range(x_dct_frames_array.shape[0]):
            real[i] = x_dct_frames_array[i]

        if channel_name == "x":
            global x_real_original_global

            x_real_original_global = real

        real_max = np.max(real)
        real_min = np.min(real)

        real = real/max(abs(real_max),abs(real_min))

        real = (real + 1)/2
        
        if self.use8bit:
            real_image = (real*255)
        else:
            real_image = (real*65535)
        if self.use8bit:
            real_image = real_image.astype(np.uint8)
        else:
            real_image = real_image.astype(np.uint16)
            
        # if channel_name == "x":
        #     max_values = np.max(imag_image, axis=0)
        #     min_values = np.min(imag_image, axis=0)
        #     mean_values = np.mean(imag_image, axis=0)  # 计算平均值

        #     columns = np.arange(imag_image.shape[1])
        #     plt.figure(figsize=(10, 5))

        #     plt.plot(columns, max_values, label='Max Values', marker='o', linestyle='-', color='r')
        #     plt.plot(columns, min_values, label='Min Values', marker='o', linestyle='-', color='b')
        #     plt.plot(columns, mean_values, label='Mean Values', marker='x', linestyle='--', color='g')  # 添加平均值

        #     plt.title('Max, Min, and Mean Values of Each Column in imag_image')
        #     plt.xlabel('Column Index')
        #     plt.ylabel('Values')
        #     plt.xticks(columns)
        #     plt.legend()

        #     plt.show()
        #     plt.close()
            
        #     plt.hist(imag_image[:,1].flatten(), bins=255, density=True,)
        #     plt.legend()
        #     plt.show()
        #     plt.close()

        # 将numpy数组转换为Pillow图像
        real_pil_image = Image.fromarray(real_image)

        real_pil_image.save(
            "data_output/{}_saveimg/{}_fft_frames_real.jp2".format(prefix, channel_name),
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[self.compression_value]  # 设置压缩率，较低的值意味着较高的压缩率
        )

        with open("data_output/{}_saveimg/{}_fft_frames.json".format(prefix,channel_name), "w") as f:
            json.dump(
                {
                    "real_max":real_max, 
                    "real_min":real_min,
                    }, 
                f)


    

        
        
    def readonechannel(self,prefix,channel_name) -> np.array:
        with open("data_output/{}_saveimg/{}_fft_frames.json".format(prefix,channel_name), "r") as f:
            json_data = json.load(f)
        real_max = json_data["real_max"]
        real_min = json_data["real_min"]
        
        real_image = imageio.imread("data_output/{}_saveimg/{}_fft_frames_real.jp2".format(prefix,channel_name))
        
        
        real_image = real_image.astype(np.float64)

        if self.use8bit:
            real_image = ((real_image)/255)
        else:
            real_image = ((real_image)/65535)
        

            
        real_image = real_image*2 - 1
        real_image = real_image*max(abs(real_max),abs(real_min))
 
        # if channel_name == "x":
        #     global x_real_original_global
        #     global x_imag_original_global
            
        #     realdiff = x_imag_original_global - imag_image
        #     plt.imshow(realdiff)
        #     plt.show()
        #     plt.close()
        #     x_imag_original_global = imag_image
        
        #reconstruct the original DCT frames
        x_reconstructed_frames = []
        for dct_frame in real_image:
            original_frame = idct(dct_frame, norm='ortho')  # 使用逆DCT并采用正交归一化
            
            x_reconstructed_frames.append(original_frame)
            
        series = np.asarray(x_reconstructed_frames)
        series = series.reshape(-1)
        return series
    
    
    
    def calculate_psnr(self,original, compressed):
        x_value = original[:, 0]
        y_value = original[:, 1]
        z_value = original[:, 2]
        #build point cloud
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(original)

        #search nearest point
        mse = 0
        distances, indices = nbrs.kneighbors(compressed)
        mse = np.mean(distances[:, 0])
            
        #calculate MSE
        # mse /= len(x_value)+len(y_value)+len(z_value)
        x_range = np.max(x_value)-np.min(x_value)
        y_range = np.max(y_value)-np.min(y_value)
        z_range = np.max(z_value)-np.min(z_value)
        max_range = pow(pow(x_range,2)+pow(y_range,2)+pow(z_range,2),0.5)
        
        
        psnr = 10*np.log10(pow(max_range,2)/mse)
        return psnr
        
    def downsample(self,x,y,z):
        x_down = np.zeros(len(x)//2)
        x_down = x[::2]
        y_down = np.zeros(len(y)//2)
        y_down = y[::2]
        z_down = np.zeros(len(z)//2)
        z_down = z[::2]
        return x_down, y_down, z_down
    
    
    def upsample(self,x,y,z):
        assert x.shape == y.shape == z.shape
        x_up = np.zeros(len(x)*2)
        y_up = np.zeros(len(y)*2)
        z_up = np.zeros(len(z)*2)
        x_up[::2] = x
        y_up[::2] = y
        z_up[::2] = z
        for i in range(1,len(x_up)-1,2):
            if (abs(x_up[i-1]-x_up[i+1])+abs(y_up[i-1]-y_up[i+1])+abs(z_up[i-1]-z_up[i+1])) > 0.01:
                x_up[i] = x_up[i-1]
                y_up[i] = y_up[i-1]
                z_up[i] = z_up[i-1]
            else:
                x_up[i] = (x_up[i-1]+x_up[i+1])/2
                y_up[i] = (y_up[i-1]+y_up[i+1])/2
                z_up[i] = (z_up[i-1]+z_up[i+1])/2
        return x_up, y_up, z_up
    
    
    def pc2mp3(self,filename):
        prefix = filename.split("/")[-1].split("_")[0]
        pcd = o3d.io.read_point_cloud(filename)
        np_pcd = np.asarray(pcd.points)
        x_value = np_pcd[:, 0]
        y_value = np_pcd[:, 1]
        z_value = np_pcd[:, 2]        
        #remove this dir 
        if os.path.exists("data_output/{}_saveimg/".format(prefix)):
            for file in os.listdir("data_output/{}_saveimg/".format(prefix)):
                os.remove("data_output/{}_saveimg/".format(prefix)+file)
            
            
        if not os.path.exists("data_output/{}_saveimg/".format(prefix)):
            os.makedirs("data_output/{}_saveimg/".format(prefix))
        
        x_value,y_value,z_value = self.downsample(x_value,y_value,z_value)
        #spilct x_value to frames, each frames has frame_size samples
        self.saveonechannel(x_value, prefix, "x")
        self.saveonechannel(y_value, prefix, "y")
        self.saveonechannel(z_value, prefix, "z")

        x_readed = self.readonechannel(prefix, "x")
        y_readed = self.readonechannel(prefix, "y")
        z_readed = self.readonechannel(prefix, "z")
        x_readed,y_readed,z_readed =  self.upsample(x_readed,y_readed,z_readed)


        pc = np.stack((x_readed, y_readed, z_readed), axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if self.visualize:
            o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("data_output/{}_saveimg/_saved_point_cloud.ply".format(prefix), pcd)
        #calculate compression ratio
        compression_size = 0
        for file in os.listdir("data_output/{}_saveimg/".format(prefix)):
            if not file.endswith(".ply"):
                compression_size += os.path.getsize("data_output/{}_saveimg/".format(prefix)+file)
        original_size = os.path.getsize("data_output/{}_saveimg/_saved_point_cloud.ply".format(prefix))

        print("original size: ", original_size)
        print("compression size: ", compression_size)
        print("compression ratio: ", original_size/compression_size)
        
        origin = np.stack((x_value, y_value, z_value), axis=-1)
        readed = np.stack((x_readed, y_readed, z_readed), axis=-1)
        psnr = self.calculate_psnr(origin, readed)
        print("PSNR: ", psnr)
        return original_size/compression_size ,psnr
    


def calc_diff(image):
    # calculate the difference along vertical direction
    diff = np.diff(image, axis=0)
    # record the initial value
    initial_value = image[0]
    return initial_value, diff

def compress(image, threshold, max_interval):
    initial_value, diff = calc_diff(image)
    # find the rows where the difference is larger than the threshold
    key_rows = np.where(np.max(np.abs(diff), axis=1) > threshold)[0] + 1
    # if there are too many rows without a key row, add a key row
    intervals = np.diff(key_rows, prepend=0)
    too_long_intervals = np.where(intervals > max_interval)[0]
    for i in reversed(too_long_intervals):
        new_key_row = key_rows[i] + max_interval // 2
        key_rows = np.insert(key_rows, i+1, new_key_row)
    # store the key_rows_data separately
    key_rows_data = image[key_rows] - image[key_rows - 1]
    diff[key_rows - 1] = 0
    # record the initial value, key_rows_data, and the differences
    compressed_data = {'initial_value': initial_value, 'diff': diff, 'key_rows': key_rows, 'key_rows_data': key_rows_data}
    return compressed_data

def decompress(compressed_data):
    initial_value = compressed_data['initial_value']
    diff = compressed_data['diff']
    key_rows = compressed_data['key_rows']
    key_rows_data = compressed_data['key_rows_data']
    # reconstruct the image from the differences, key_rows_data, and the initial value
    image = np.empty_like(diff)
    image[0] = initial_value
    for i, key_row in enumerate(key_rows):
        if i == 0:
            image[:key_row] = np.cumsum(diff[:key_row], axis=0) + initial_value
        else:
            image[key_rows[i-1]:key_row] = np.cumsum(diff[key_rows[i-1]:key_row], axis=0) + image[key_rows[i-1]-1]
        # add the key_rows_data back to the corresponding row
        image[key_row-1] += key_rows_data[i]
    image[key_rows[-1]:] = np.cumsum(diff[key_rows[-1]:], axis=0) + image[key_rows[-1]-1]
    return image


path = "./data_input/000000.ply"

if __name__ == "__main__":

    frame_sizes = [4,8,16,32,64,128, 256, 512,1024,2048]
    cp_levels = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30]
    ratios = np.zeros((len(frame_sizes), len(cp_levels)))
    psnrs = np.zeros((len(frame_sizes), len(cp_levels)))
    i = 0
    j = 0
    for  frame_size in frame_sizes:
        for  cplevel in cp_levels:
            print("-----------------")  
            print("frame_size: ", frame_size)
            print("cplevel: ", cplevel)
            pcc = PCcompression(frame_size,cplevel,0,False,False)
            ratios[i,j], psnrs[i,j] = pcc.pc2mp3(path)
            print("-----------------")
            j+=1
        j = 0
        i+=1
    np.save("ratios.npy", ratios)
    np.save("psnrs.npy", psnrs)
    plt.imshow(psnrs)
    plt.xticks(np.arange(len(cp_levels)), labels=cp_levels)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    plt.show()
    plt.close()
    print("Done.")