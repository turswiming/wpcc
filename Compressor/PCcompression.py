"""
writen by: Ziqi Li

This file is used to compress point cloud data using fft and image compression method.

The main function is pc2mp3, which takes a path of a point cloud file as input, and output the compression ratio and PSNR of the compression.

usage:

    pcc = PCcompression(32,1,0,10/10,0,True, True, False)
    pcc.pc2mp3(path, "./data_output/01_save")
    params:
        self,
        frame_size,
        compression_value,
        highres_rate=0,
        Ocbit_threshold=1/10, 
        overlap_size = 0,
        dodownsample=False,
        visualize=False,
        use8bit=False
    
"""

import glymur
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
from getScaleParameter import getScaleParameter as gsp
from sklearn.neighbors import NearestNeighbors
import numpy as np
import Compressor.framewidthTable as fwt
from bitarray import bitarray
from scipy.optimize import minimize
import math
from Compressor.datatype.SVD import SVD_Data
x_real_original_global = np.array([])
ununiQuantizeNum = 2

class PCcompression:
    def __init__(self,
                 frame_size,
                 compression_value,
                 dodownsample=False,
                 visualize=False
                 ) -> None:
        if frame_size % 2 != 0:
            raise ValueError("frame_size should be even")
        fwtable = fwt.FrameSizeTable()
        self.frame_size = frame_size
        self.compression_value = compression_value
        self.dodownsample = dodownsample
        self.visualize = visualize
        self.tiny = 10000
        pass



    def is_occluded(self,center, points, threshold=0.1,ballsize=0.003):
        print("center: ", center)
        #convert points-conter to polar coordinates
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        distances = np.linalg.norm(points - center, axis=1)
        distances = distances / np.max(distances) *3.14
        #calculate pitch and yaw
        pitchs = np.arctan2(points[:, 2] - center[2], distances)
        #knn search
        #build up knn
        # angle_pitch= np.stack((angles, pitchs), axis=-1)
        # knn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(angle_pitch)
        # #search nearest point
        # distances, indices = knn.kneighbors(angle_pitch)
        # repulsions = np.zeros((len(angles), 1))
        # for i in range(0,len(angles),100):
        #     repulsions[i] =   max(10,1/np.sum(distances[1]))
        angle_pitch_map = {}
        occluded_number = 0
        for i in range(len(angles)):
            tuple = (int(angles[i]/ballsize), int(pitchs[i]/ballsize))
            if tuple not in angle_pitch_map:
                angle_pitch_map[tuple] = 1
            else:
                occluded_number +=1

        # angle_pitch_array = np.zeros((int(3.14*2/ballsize+1), int(3.14*2/ballsize+1)))
        # for i in range(len(angles)):
        #     angle_pitch_array[int(angles[i]/ballsize), int(pitchs[i]/ballsize)] = 1
        #show the map
        # plt.imshow(angle_pitch_array)
        # plt.show()
        # plt.close()
        print("occluded_number: ", occluded_number)
        # print("repulsions: ", np.sum(repulsions))
        return occluded_number

    def estimate_lidar_position(self,pcd):
        np_pcd = np.asarray(pcd.points)
        bounding_box = pcd.get_axis_aligned_bounding_box()
        initial_guess = np.array([0, 0, 0])
        is_occluded = self.is_occluded
        def objective_function_first_step(params):
            return is_occluded(params, np_pcd)

        result = minimize(
            objective_function_first_step, 
            initial_guess,
            method='BFGS',
            bounds=[(-1, 1), (-1, 1), (-1, 1)],
            options={"maxiter": 100}
            )
        
        return result.x
    def __calc_diff(self, image):
        # calculate the difference along vertical direction
        diff = np.diff(image, axis=0)
        # record the initial value
        initial_value = image[0]
        return initial_value, diff

    def __dft(self, x):
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

    def __get_min_max(self, image, width, height):
        min = np.zeros((math.ceil(image.shape[0] / width), math.ceil(image.shape[1] / height)))
        max = np.zeros((math.ceil(image.shape[0] / width), math.ceil(image.shape[1] / height)))
        for i in range(0, image.shape[0], width):
            for j in range(0, image.shape[1], height):
                min[int(i / width), int(j / height)] = np.min(image[i:i + width, j:j + height])
                max[int(i / width), int(j / height)] = np.max(image[i:i + width, j:j + height])
        return min, max

    def __sintransform(self, image, level):
        if level == 0:
            return image
        else:
            image = np.sin(image * 3.1415926 / 2)
            return self.__sintransform(image, level - 1)

    def __arcsintransform(self, image, level):
        if level == 0:
            return image
        else:
            image = np.arcsin(image) / (3.1415926 / 2)
            return self.__arcsintransform(image, level - 1)

    def __arctantransform(self, image, level):
        if level == 0:
            return image
        else:
            image = np.arctan(image / (3.1415926 / 2))
            return self.__arctantransform(image, level - 1)

    def __tantransform(self, image, level):
        if level == 0:
            return image
        else:
            image = np.tan(image) * 3.1415926 / 2
            return self.__tantransform(image, level - 1)

    def __UnuniQuantize(self, image: np.array, level: float):
        # return self.__arcsintransform(image, level)
        # return self.__arctantransform(image, level)
        if level == 0:
            return image
        return np.arctan(image * level) * 2 / 3.1415926
        return self.__sintransform(image, level)
        pass

    def __unpackUnuniQuantize(self, image: np.array, level: float):
        # return self.__sintransform(image, level)
        # return self.__tantransform(image, level)
        if level == 0:
            return image
        return np.tan((image) * 3.1415926 / 2) / level
        return self.__arcsintransform(image, level)
        pass

    def genlowPrecisionPic(self, image: np.array, threshold: float):

        mask = (image > -threshold) & (image < threshold)
        print("mask: ", mask[mask == True].shape)
        print("mask: ", mask[mask == False].shape)
        image_clamped = np.clip(image, -threshold, threshold)
        image_scaled = image_clamped / threshold

        image_scaled = image_clamped / threshold

        mask_line = mask.reshape(-1)
        false_indices = np.where(mask_line == False)[0]
        return image_scaled, false_indices

    def rebuildHighPrecisionPic(self, lowPrecisionPic: np.array, highPrecisionNumbers: np.array, mask_line: np.array,
                                threshold: float):
        image = lowPrecisionPic * threshold
        for index in range(mask_line.shape[0]):
            indices = mask_line[index]
            x = indices // image.shape[1]
            y = indices % image.shape[1]
            image[x, y] = highPrecisionNumbers[index]
            x = indices // image.shape[1]
            y = indices % image.shape[1]
            image[x, y] = highPrecisionNumbers[index]
        return image

    def __DCTProcess(self, value, channel_name) -> np.array:
        # 1.1 cliping
        x_frames = []
        for i in range(0, len(value), self.frame_size):
            x_frames.append(value[i:i + self.frame_size])
        # 1.2 DCT
        # apply DCT to each frame
        x_dct_frames = []
        for frame in x_frames:
            if len(frame) < self.frame_size:
                continue
            dct_result = dct(frame, norm='ortho')
            x_dct_frames.append(dct_result)


        x_dct_frames_array = np.array(x_dct_frames)
        real = np.zeros(x_dct_frames_array.shape)
        for i in range(x_dct_frames_array.shape[0]):
            real[i] = x_dct_frames_array[i]

        if channel_name == "x":
            global x_real_original_global

            x_real_original_global = real
        return real

    def __createContinuedBitmap(self, bitmap: np.array) -> np.array:
        bitmap = bitmap.astype(np.float64)
        #blur the bitmap
        for i in range(1, bitmap.shape[0] - 1):
            bitmap[i] = (bitmap[i - 1] + bitmap[i] + bitmap[i + 1]) / 3
        bitmap[1] = (bitmap[0] + bitmap[1] + bitmap[2]) / 3
        bitmap[bitmap.shape[0] - 2] = (bitmap[bitmap.shape[0] - 3] + bitmap[bitmap.shape[0] - 2] + bitmap[bitmap.shape[0] - 1]) / 3
        
        new_bitmap = np.zeros(bitmap.shape[0])
        for i in range(bitmap.shape[0]):
            if bitmap[i] >0.5:
                new_bitmap[i] = 1
        
        bitmap = bitmap.astype(np.bool)
        return bitmap


    def __saveDCTFrames(self, savedir, x_image, y_image, z_image):
        global combined_image
        combined_image = np.stack((x_image, y_image, z_image), axis=-1)

        metadata = {}


        max_values = np.max(combined_image)
        min_values = np.min(combined_image)

        combined_image = combined_image / max(abs(max_values), abs(min_values))
        combined_image = self.__UnuniQuantize(combined_image, ununiQuantizeNum)
        combined_image = (combined_image + 1) / 2
        combined_image = (combined_image * 65535)
        combined_image = combined_image.astype(np.uint16)
        jp2_filename = "{}/dct_frames_right.jp2".format(savedir)
        tile_size = (combined_image.shape[0], combined_image.shape[1])
        jp2 = glymur.Jp2k(
            jp2_filename,
            data=combined_image,
            numres=1,
            cratios=(self.compression_value,),
            tilesize=tile_size,
            display_resolution=None,
            modesw=1,
            mct=False,
            # remove unused markers
            eph=False,
            plt=False,
            sop=False,
            tlm=False,
        )
        metadata["max_values"] = max_values
        metadata["min_values"] = min_values

        metadata["Downsample"] = 1 if self.dodownsample else 0
        metadata["FrameSize"] = self.frame_size
        with open("{}/metadata.json".format(savedir), "w") as f:
            json.dump(metadata, f)
        pass

    def __saveBitArray(self, bitarray: bitarray, path: str):
        with open(path, "wb") as f:
            bitarray.tofile(f)
        pass

    def __readdata(self, savedir) -> np.array:
        with open("{}/metadata.json".format(savedir), "r") as f:
            metadata = json.load(f)

        max_values = metadata["max_values"]
        min_values = metadata["min_values"]
        if os.path.exists("{}/dct_frames_right.jp2".format(savedir)):
            jp2k_high = glymur.Jp2k(
                "{}/dct_frames_right.jp2".format(savedir),
            )
            image = jp2k_high[:]
            image = image.astype(np.float64)
            image = ((image) / 65535)
            image = image * 2 - 1
            wide = image.shape[1]

        
        image = self.__unpackUnuniQuantize(image, ununiQuantizeNum)
        image = image * max(abs(max_values), abs(min_values))


        return image[:, :, 0], image[:, :, 1], image[:, :, 2], metadata

        # reconstruct the original DCT frames

    def __IDCTProcess(self, real_image, channel_name) -> np.array:
        if channel_name == "x":
            # global combined_image

            # realdiff = combined_image[:,:,0] - real_image
            # plt.plot(real_image[1])
            # plt.plot(combined_image[1,:,0])
            # # plt.imshow(realdiff[:50,:])
            # plt.show()
            # plt.close()
            x_imag_original_global = real_image

        x_reconstructed_frames = []
        for id ,dct_frame in enumerate(real_image):
            original_frame = idct(dct_frame, norm='ortho')  # 使用逆DCT并采用正交归一化
            x_reconstructed_frames.append(original_frame)

        overlap_frames = np.asarray(x_reconstructed_frames)
        series = overlap_frames.reshape(-1)
        return series
        

    def __calculate_psnr(self, original, compressed):
        x_value = original[:, 0]
        y_value = original[:, 1]
        z_value = original[:, 2]
        x_range = np.max(x_value) - np.min(x_value)
        y_range = np.max(y_value) - np.min(y_value)
        z_range = np.max(z_value) - np.min(z_value)
        max_range = pow(pow(x_range, 2) + pow(y_range, 2) + pow(z_range, 2), 0.5)


        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(original)
        mse = 0
        _, indices = nbrs.kneighbors(compressed)
        distances1 = np.zeros((len(indices), 1))
        for i in range(len(indices)):
            distances1[i] = np.linalg.norm(original[indices[i][0]] - compressed[i])
        mse = np.mean(np.square(distances1))
        psnr1 = 10 * np.log10(pow(max_range, 2) / mse)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(compressed)
        mse = 0
        _, indices = nbrs.kneighbors(original)
        distances2 = np.zeros((len(indices), 1))
        for i in range(len(indices)):
            distances2[i] = np.linalg.norm(compressed[indices[i][0]] - original[i])
        mse = np.mean(np.square(distances2))
        psnr2 = 10 * np.log10(pow(max_range, 2) / mse)

        return (psnr1+psnr2)/2 ,distances1

    def __downsample(self, x, y, z):
        x_down = np.zeros(len(x) // 2)
        x_down = x[::2]
        y_down = np.zeros(len(y) // 2)
        y_down = np.zeros(len(y) // 2)
        y_down = y[::2]
        z_down = np.zeros(len(z) // 2)
        z_down = np.zeros(len(z) // 2)
        z_down = z[::2]
        return x_down, y_down, z_down

    def __upsample(self, x, y, z):
        assert x.shape == y.shape == z.shape
        x_up = np.zeros(len(x) * 2)
        y_up = np.zeros(len(y) * 2)
        z_up = np.zeros(len(z) * 2)
        x_up = np.zeros(len(x) * 2)
        y_up = np.zeros(len(y) * 2)
        z_up = np.zeros(len(z) * 2)
        x_up[::2] = x
        y_up[::2] = y
        z_up[::2] = z
        for i in range(1, len(x_up) - 1, 2):
            distance = abs(x_up[i - 1] - x_up[i + 1]) + abs(y_up[i - 1] - y_up[i + 1]) + abs(
                z_up[i - 1] - z_up[i + 1])
            if distance > 0.01:
                x_up[i] = x_up[i - 1]
                y_up[i] = y_up[i - 1]
                z_up[i] = z_up[i - 1]
            else:
                x_up[i] = (x_up[i - 1] + x_up[i + 1]) / 2
                y_up[i] = (y_up[i - 1] + y_up[i + 1]) / 2
                z_up[i] = (z_up[i - 1] + z_up[i + 1]) / 2
                x_up[i] = (x_up[i - 1] + x_up[i + 1]) / 2
                y_up[i] = (y_up[i - 1] + y_up[i + 1]) / 2
                z_up[i] = (z_up[i - 1] + z_up[i + 1]) / 2
        return x_up, y_up, z_up
    def __SVD(self, pc:np.array)->SVD_Data:
        #split pc into frames with frame_size samples
        Us = []
        Ss = []
        Vs = []
        for i in range(0, pc.shape[0], self.frame_size):
            pc_frame = pc[i:i + self.frame_size]
            if len(pc_frame) < self.frame_size:
                continue
            U, S, V = self.__SVD_single(pc_frame)
            Us.append(U)
            Ss.append(S)
            Vs.append(V)
        Us = np.array(Us)
        Ss = np.array(Ss)
        Vs = np.array(Vs)
        U_max = np.max(Us)
        U_min = np.min(Us)
        S_max = np.max(Ss)
        S_min = np.min(Ss)
        V_max = np.max(Vs)
        V_min = np.min(Vs)
        Us_range = max(abs(U_max), abs(U_min))
        Ss_range = max(abs(S_max), abs(S_min))
        Vs_range = max(abs(V_max), abs(V_min))

        Us = Us / Us_range
        Ss = Ss / Ss_range
        Vs = Vs / Vs_range
        Us = (Us*128).astype(np.int8).astype(np.float64)/128
        if self.visualize:
            print(Ss[0].shape)
            from Compressor.visualize.show_matrix import show_matrix, show_histogram
            show_histogram(Vs)
            showed = Vs
            matrix = np.zeros((len(showed), showed[0].shape[0]*showed[0].shape[1]))
            for i in range(len(showed)):
                matrix[i] = showed[i].reshape(-1)
            print(matrix.shape)
            show_matrix(matrix)
        return SVD_Data(Us, Ss, Vs, Us_range, Ss_range, Vs_range)

    def __Reverse_SVD(self, svd_data:SVD_Data)->np.array:
        Us, Ss, Vs,Us_range,Ss_range,Vs_range = svd_data.get()
        Us = Us * Us_range
        Ss = Ss * Ss_range
        Vs = Vs * Vs_range
        pc = np.zeros((len(Us) * self.frame_size, 3))
        for i in range(len(Us)):
            U = Us[i]
            S = Ss[i]
            V = Vs[i]
            pc[i * self.frame_size:(i + 1) * self.frame_size] = self.__Reverse_SVD_single(U, S, V)
        return pc


    def __SVD_single(self, pc)->tuple[np.array, np.array, np.array]:
        # truncated SVD with k=1
        from scipy.sparse.linalg import svds
        pc = pc.transpose() # we need to explain why
        U, S, V = svds(pc, k=2)
        return U, S, V
    def __Reverse_SVD_single(self, U, S, V):
        return np.dot(U, np.dot(np.diag(S), V)).transpose()
    def pc2mp3(self, filename, savedir):

        pcd = o3d.io.read_point_cloud(filename)
        np_pcd = np.asarray(pcd.points)
        #detect the center of the point cloud
        # center = self.estimate_lidar_position(pcd)
        # print("center: ", center)
        x_value = np_pcd[:, 0]
        y_value = np_pcd[:, 1]
        z_value = np_pcd[:, 2]



        # remove this directory
        if os.path.exists(savedir):
            for file in os.listdir(savedir):
                os.remove("{}/".format(savedir) + file)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if self.dodownsample:
            x_value, y_value, z_value = self.__downsample(x_value, y_value, z_value)
        # pc = np.stack((x_value, y_value, z_value), axis=-1)
        # svd_data = self.__SVD(pc)
        
        # spilct x_value to frames, each frames has frame_size samples
        x_image = self.__DCTProcess(x_value, "x")
        y_image = self.__DCTProcess(y_value, "y")
        z_image = self.__DCTProcess(z_value, "z")
        # save DCT frames
        self.__saveDCTFrames(savedir, x_image, y_image, z_image)
        # ---------------------------------------------------------
        # above is saver

        # here is reader
        # ---------------------------------------------------------

        x_read_image, y_read_image, z_read_image, metadata = self.__readdata(savedir)
        x_readed = self.__IDCTProcess(x_read_image, "x")
        y_readed = self.__IDCTProcess(y_read_image, "y")
        z_readed = self.__IDCTProcess(z_read_image, "z")
        # pc = self.__Reverse_SVD(svd_data)
        # x_readed = pc[:, 0]
        # y_readed = pc[:, 1]
        # z_readed = pc[:, 2]
        if metadata["Downsample"] == 1:
            x_readed, y_readed, z_readed = self.__upsample(x_readed, y_readed, z_readed)
        
        pc = np.stack((x_readed, y_readed, z_readed), axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        o3d.io.write_point_cloud("{}/saved_point_cloud.ply".format(savedir), pcd)
        print("saved point cloud saved at: ", "{}/saved_point_cloud.ply".format(savedir))
        pc_original = np.stack((x_value, y_value, z_value), axis=-1) 
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(pc_original)
        o3d.io.write_point_cloud("{}/original_point_cloud.ply".format(savedir), pcd_original)
        print("original point cloud saved at: ", "{}/original_point_cloud.ply".format(savedir))
        # calculate compression ratio
        compression_size = 0
        for file in os.listdir("{}/".format(savedir)):
            if not file.endswith(".ply"):
                compression_size += os.path.getsize("{}/".format(savedir) + file)
        original_size = os.path.getsize("{}/saved_point_cloud.ply".format(savedir))

        print("original size: ", original_size)
        print("compression size: ", compression_size)
        print("compression ratio: ", original_size / compression_size)

        origin = np.stack((x_value, y_value, z_value), axis=-1)
        readed = np.stack((x_readed, y_readed, z_readed), axis=-1)
        psnr,distances = self.__calculate_psnr(origin, readed)
        print("BPP: ", 8 * compression_size / x_readed.shape[0])
        print("PSNR: ", psnr)
        color = np.zeros((x_readed.shape[0], 3))
        print(distances.shape)
        distances = distances / np.max(distances)
        distances = distances
        blue_color = np.array([0, 0, 1])
        red_color = np.array([1, 0, 0])
        for i in range(distances.shape[0]):
            color[i] = distances[i] * red_color + (1 - distances[i]) * blue_color
        pcd.colors = o3d.utility.Vector3dVector(color)
        #draw a big ball mesh in the center

        if self.visualize:
            o3d.visualization.draw_geometries([pcd])
            # distancetocenter = np.linalg.norm(readed, axis=1)
            # plt.scatter(distancetocenter, distances[:, 0])
            # plt.xlabel('Distance to Center')
            # plt.ylabel('Distances')
            # plt.title('Scatter Plot of Distances')
            # plt.show()
            # plt.close()
        return 8 * compression_size / x_readed.shape[0], psnr
