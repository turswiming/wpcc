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
import scipy.fftpack as fft
import struct
import zlib
import open3d as o3d
import os
import json
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt
import math
from pydub import AudioSegment
x_real_original_global = np.array([])
x_imag_original_global = np.array([])
class PCcompression:
    def __init__(self,box_size,frame_size,compression_value ,visualize = False, use8bit =False) -> None:
        self.box_size = box_size
        self.frame_size = frame_size
        self.compression_value = compression_value
        self.visualize = visualize
        self.use8bit = use8bit
        pass

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

    def saveonechannel(self,value, prefix,channel_name) ->np.array:
        #1.1 cliping
        x_frames = []
        for i in range(0, len(value), self.frame_size):
            x_frames.append(value[i:i+self.frame_size])
        #1.2 fft
        #apply fft to each frame
        x_fft_frames = []
        max_length = 0
        for frame in x_frames:
            fft_result = fft.fft(frame)
            if len(fft_result) != self.frame_size:
                break
            x_fft_frames.append(fft_result)
            if len(fft_result) > max_length:
                max_length = len(fft_result)
        
        x_fft_frames_array = np.array(x_fft_frames)
        real = np.zeros(x_fft_frames_array.shape)
        imag = np.zeros(x_fft_frames_array.shape)
        for i in range(x_fft_frames_array.shape[0]):
            real[i] = x_fft_frames_array[i].real
            imag[i] = x_fft_frames_array[i].imag
        #2.1 store max min values
        real_max = np.max(real)
        real_min = np.min(real)
        imag_max = np.max(imag)
        imag_min = np.min(imag)
        with open("data_output/{}_saveimg/{}_fft_frames.json".format(prefix,channel_name), "w") as f:
            json.dump({"real_max":real_max, "real_min":real_min, "imag_max":imag_max, "imag_min":imag_min}, f)

        real = (real-np.min(real))/(np.max(real)-np.min(real))
        imag = (imag-np.min(imag))/(np.max(imag)-np.min(imag))
        real = pow(real, 0.5)
        imag = pow(imag, 0.5)
        if self.use8bit:
            real_image = (real*255)
            imag_image = (imag*255)
        else:
            real_image = (real*65535)
            imag_image = (imag*65535)
        if channel_name == "x":
            global x_real_original_global
            global x_imag_original_global
            x_real_original_global = real_image
            x_imag_original_global = imag_image
        # #save to image
        # # 获取数组每个64*64方格内最大值最小值
        # real_min_map, real_max_map = self.get_min_max(real_image, self.box_size, self.box_size)
        # imag_min_map, imag_max_map = self.get_min_max(imag_image, self.box_size, self.box_size)

        # real_min_map_adjusted = np.repeat(real_min_map, self.box_size,axis=0)  # Adjusting the shape
        # real_max_map_adjusted = np.repeat(real_max_map, self.box_size,axis=0)
        # imag_min_map_adjusted = np.repeat(imag_min_map, self.box_size,axis=0)
        # imag_max_map_adjusted = np.repeat(imag_max_map, self.box_size,axis=0)

        # real_min_map_adjusted = np.repeat(real_min_map_adjusted, self.box_size,axis=1)  # Adjusting the shape
        # real_max_map_adjusted = np.repeat(real_max_map_adjusted, self.box_size,axis=1)
        # imag_min_map_adjusted = np.repeat(imag_min_map_adjusted, self.box_size,axis=1)
        # imag_max_map_adjusted = np.repeat(imag_max_map_adjusted, self.box_size,axis=1)
        # # cut real_min_map_adjusted to the shape of real_image
        # real_min_map_adjusted = real_min_map_adjusted[:real_image.shape[0], :real_image.shape[1]]
        # real_max_map_adjusted = real_max_map_adjusted[:real_image.shape[0], :real_image.shape[1]]
        # imag_min_map_adjusted = imag_min_map_adjusted[:imag_image.shape[0], :imag_image.shape[1]]
        # imag_max_map_adjusted = imag_max_map_adjusted[:imag_image.shape[0], :imag_image.shape[1]]
        
        # real_image = (real_image - real_min_map_adjusted) / (real_max_map_adjusted - real_min_map_adjusted) * 65535
        # imag_image = (imag_image - imag_min_map_adjusted) / (imag_max_map_adjusted - imag_min_map_adjusted) * 65535
        # for i in range(real_image.shape[1]):
        #     wave = real_image[:,i]
        #     wave = wave-32767
        #     wave = wave.astype(np.int16)
        #     audio = AudioSegment(wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        #     audio.export("data_output/{}_saveimg/{}_fft_frames_real_{}.wav".format(prefix,channel_name,i), format="wav")
            
        # for i in range(imag_image.shape[1]):
        #     wave = imag_image[:,i]
        #     wave = wave-32767            
        #     wave = wave.astype(np.int16)
        #     audio = AudioSegment(wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        #     audio.export("data_output/{}_saveimg/{}_fft_frames_imag_{}.wav".format(prefix,channel_name,i), format="wav") 
            
    
        if self.use8bit:
            real_image = real_image.astype(np.uint8)
            imag_image = imag_image.astype(np.uint8)
        else:
            real_image = real_image.astype(np.uint16)
            imag_image = imag_image.astype(np.uint16)
        # 将numpy数组转换为Pillow图像
        real_pil_image = Image.fromarray(real_image)
        imag_pil_image = Image.fromarray(imag_image)

        # real_min_map = real_min_map.astype(np.uint16)
        # real_max_map = real_max_map.astype(np.uint16)
        # imag_min_map = imag_min_map.astype(np.uint16)
        # imag_max_map = imag_max_map.astype(np.uint16)
        # # 保存最大最小图像
        # real_pil_min_image = Image.fromarray(real_min_map)
        # real_pil_max_image = Image.fromarray(real_max_map)
        # imag_pil_min_image = Image.fromarray(imag_min_map)
        # imag_pil_max_image = Image.fromarray(imag_max_map)
        # real_pil_min_image.save("data_output/{}_saveimg/{}_fft_frames_real_min.png".format(prefix, channel_name))
        # real_pil_max_image.save("data_output/{}_saveimg/{}_fft_frames_real_max.png".format(prefix, channel_name))
        # imag_pil_min_image.save("data_output/{}_saveimg/{}_fft_frames_imag_min.png".format(prefix, channel_name))
        # imag_pil_max_image.save("data_output/{}_saveimg/{}_fft_frames_imag_max.png".format(prefix, channel_name))


        real_pil_image.save(
            "data_output/{}_saveimg/{}_fft_frames_real.jp2".format(prefix, channel_name),
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[self.compression_value]  # 设置压缩率，较低的值意味着较高的压缩率
        )

        # 保存imag部分的图像
        imag_pil_image.save(
            "data_output/{}_saveimg/{}_fft_frames_imag.jp2".format(prefix, channel_name),
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[self.compression_value]  # 设置压缩率
        )

        # 保存imag部分的图像
        
        # real_pil_image.save(
        #     "data_output/{}_saveimg/{}_fft_frames_real.jpg".format(prefix, channel_name),
        #     format="JPEG",
        #     quality=85,  # 设置压缩率
        # )
            
        # imag_pil_image.save(
        #     "data_output/{}_saveimg/{}_fft_frames_imag.jpg".format(prefix, channel_name),
        #     format="JPEG",
        #     quality=85,  # 设置压缩率
        # )
        # plt.hist(real_image.flatten(), bins=255, color='r')
        # plt.show()
        # plt.close()
        # real_pil_image.save(
        #     "data_output/{}_saveimg/{}_fft_frames_real.png".format(prefix,channel_name),
        #     format="png",
        #     )
        # imag_pil_image.save(
        #     "data_output/{}_saveimg/{}_fft_frames_imag.png".format(prefix,channel_name),
        #     format="png",
        #     )
        


        
        
    def readonechannel(self,prefix,channel_name) -> np.array:
        with open("data_output/{}_saveimg/{}_fft_frames.json".format(prefix,channel_name), "r") as f:
            json_data = json.load(f)
        real_max = json_data["real_max"]
        real_min = json_data["real_min"]
        imag_max = json_data["imag_max"]
        imag_min = json_data["imag_min"]
        real_image = imageio.imread("data_output/{}_saveimg/{}_fft_frames_real.jp2".format(prefix,channel_name))
        imag_image = imageio.imread("data_output/{}_saveimg/{}_fft_frames_imag.jp2".format(prefix,channel_name))
        # real_waves = {}
        # imag_waves = {}
        # for files in os.listdir("data_output/{}_saveimg/".format(prefix)):
        #     if files.endswith(".wav"):
        #         audio = AudioSegment.from_file("data_output/{}_saveimg/".format(prefix)+files,format="wav")
        #         np_audio = np.array(audio.get_array_of_samples())
        #         index = int(files.split("_")[-1].split(".")[0])
        #         np_audio = np_audio.astype(np.float64)
        #         np_audio = np_audio + 32767
        #         if files.startswith("x_fft_frames_real"):
        #             real_waves[index] = np_audio
        #         elif files.startswith("x_fft_frames_imag"):
        #             imag_waves[index] = np_audio
        
        # real_image = np.zeros((len(real_waves[0]),len(real_waves)))
        # imag_image = np.zeros((len(imag_waves[0]),len(imag_waves)))
        # for key in real_waves:
        #     real_image[:,key] = real_waves[key]
        # for key in imag_waves:
        #     imag_image[:,key] = imag_waves[key]

        # if channel_name == "x":
        #     diff = x_imag_original_global-imag_image
        #     original_avg = np.mean(x_imag_original_global)
        #     diff_avg = np.mean(diff)
        #     now_avg = np.mean(imag_image)
        #     print("original_avg: ", original_avg)
        #     print("diff_avg: ", diff_avg)
        #     print("now_avg: ", now_avg)
        #     plt.imshow(diff)
        #     plt.show()
        #     plt.close()
        # real_max_map = imageio.imread("data_output/{}_saveimg/{}_fft_frames_real_max.png".format(prefix,channel_name))
        # real_min_map = imageio.imread("data_output/{}_saveimg/{}_fft_frames_real_min.png".format(prefix,channel_name))
        # imag_max_map = imageio.imread("data_output/{}_saveimg/{}_fft_frames_imag_max.png".format(prefix,channel_name))
        # imag_min_map = imageio.imread("data_output/{}_saveimg/{}_fft_frames_imag_min.png".format(prefix,channel_name))

        #     # 根据最大值最小值进行压缩
        # real_min_map_adjusted = np.repeat(real_min_map, self.box_size,axis=0)  # Adjusting the shape
        # real_max_map_adjusted = np.repeat(real_max_map, self.box_size,axis=0)
        # imag_min_map_adjusted = np.repeat(imag_min_map, self.box_size,axis=0)
        # imag_max_map_adjusted = np.repeat(imag_max_map, self.box_size,axis=0)

        # real_min_map_adjusted = np.repeat(real_min_map_adjusted, self.box_size,axis=1)  # Adjusting the shape
        # real_max_map_adjusted = np.repeat(real_max_map_adjusted, self.box_size,axis=1)
        # imag_min_map_adjusted = np.repeat(imag_min_map_adjusted, self.box_size,axis=1)
        # imag_max_map_adjusted = np.repeat(imag_max_map_adjusted, self.box_size,axis=1)

        # # cut real_min_map_adjusted to the shape of real_image
        # real_min_map_adjusted = real_min_map_adjusted[:real_image.shape[0], :real_image.shape[1]]
        # real_max_map_adjusted = real_max_map_adjusted[:real_image.shape[0], :real_image.shape[1]]
        # imag_min_map_adjusted = imag_min_map_adjusted[:imag_image.shape[0], :imag_image.shape[1]]
        # imag_max_map_adjusted = imag_max_map_adjusted[:imag_image.shape[0], :imag_image.shape[1]]
        # real_image = real_image.astype(np.float64)
        # imag_image = imag_image.astype(np.float64)
        
        # real_image = real_image * (real_max_map_adjusted - real_min_map_adjusted) / 65535 + real_min_map_adjusted
        # imag_image = imag_image * (imag_max_map_adjusted - imag_min_map_adjusted) / 65535 + imag_min_map_adjusted
        
        
        
        real_image = real_image.astype(np.float64)
        imag_image = imag_image.astype(np.float64)
        if self.use8bit:
            real_image = ((real_image)/255)
            imag_image = ((imag_image)/255)
        else:
            real_image = ((real_image)/65535)
            imag_image = ((imag_image)/65535)
        real_image  =pow(real_image, 1/0.5)
        imag_image  =pow(imag_image, 1/0.5)
        real_image = real_image*(real_max-real_min) + real_min
        imag_image = imag_image*(imag_max-imag_min) + imag_min
        fft_frames = real_image + 1j*imag_image
        #reconstruct the original fft frames
        x_original_frames = []
        for fft_frame in fft_frames:
            original_frame = np.fft.ifft(fft_frame)
            
            original_frame = np.real(original_frame)
            
            x_original_frames.append(original_frame)

        return np.concatenate(x_original_frames)

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
        
        
        #spilct x_value to frames, each frames has frame_size samples
        self.saveonechannel(x_value, prefix, "x")
        self.saveonechannel(y_value, prefix, "y")
        self.saveonechannel(z_value, prefix, "z")

        x_readed = self.readonechannel(prefix, "x")
        y_readed = self.readonechannel(prefix, "y")
        z_readed = self.readonechannel(prefix, "z")
        


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
        
        #calculate MSE
        mse = 0
        mse += np.sum(np.square(x_value[:x_readed.shape[0]]-x_readed))
        mse += np.sum(np.square(y_value[:x_readed.shape[0]]-y_readed))
        mse += np.sum(np.square(z_value[:x_readed.shape[0]]-z_readed))
        mse /= len(x_value)+len(y_value)+len(z_value)
        x_range = np.max(x_value)-np.min(x_value)
        y_range = np.max(y_value)-np.min(y_value)
        z_range = np.max(z_value)-np.min(z_value)
        range = pow(pow(x_range,2)+pow(y_range,2)+pow(z_range,2),0.5)
        
        
        psnr = 10*np.log10(pow(range,2)/mse)
        
        print("PSNR: ", psnr)
        return original_size/compression_size ,psnr
    


path = "./data_input/0001_point_cloud.ply"

if __name__ == "__main__":
    pcc = PCcompression(4096, 4,1,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 4,2,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 4,3,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 4,5,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 4,10,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 4,20,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 2048,50,False,False)
    pcc.pc2mp3(path)
    pcc = PCcompression(4096, 2048,100,False,False)
    pcc.pc2mp3(path)
    box_sizes = [128,256,512,1024]
    frame_sizes = [4,8,16,32,64,128, 256, 512,1024,2048]
    compression_values = [1,2,3,4,5,6,7,8,9,10]
    ratios = np.zeros((len(frame_sizes), len(compression_values)))
    psnrs = np.zeros((len(frame_sizes), len(compression_values)))
    i = 0
    j = 0
    for  frame_size in frame_sizes:
        for  conpression_value in compression_values:

            pcc = PCcompression(1024, frame_size,conpression_value)
            print("frame_size: ", frame_size, "conpression_value: ", conpression_value)
            ratios[i,j], psnrs[i,j] = pcc.pc2mp3(path)
            print("-----------------")
            j+=1
        j = 0
        i+=1
    np.save("ratios.npy", ratios)
    np.save("psnrs.npy", psnrs)
    plt.imshow(ratios)
    plt.xticks(np.arange(len(compression_values)), labels=compression_values)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)

    plt.show()
    plt.close()
    plt.imshow(psnrs)
    plt.show()
    plt.close()
    print("Done.")