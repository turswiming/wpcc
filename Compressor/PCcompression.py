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

x_real_original_global = np.array([])
ununiQuantizeNum = 2

class PCcompression:
    def __init__(self,
                 frame_size,
                 compression_value,
                 highres_rate=0,
                 Ocbit_threshold=1/10,
                 overlap_size = 0,
                 dodownsample=False,
                 visualize=False,
                 use8bit=False) -> None:
        if frame_size % 2 != 0:
            raise ValueError("frame_size should be even")
        fwtable = fwt.FrameSizeTable()
        # if frame_size not in fwtable.get_frame_sizes(compression_value):
        #     print("frame_size: ", frame_size, " is not suggested, please use the following frame sizes: ",
        #           fwtable.get_frame_sizes(compression_value))
        #     print("frame_size will be set to", fwtable.get_frame_sizes(compression_value)[0],
        #           " automatically this time")
        #     self.frame_size = fwtable.get_frame_sizes(compression_value)[0]
        # else:
        self.frame_size = frame_size
        self.compression_value = compression_value
        self.overlap_size = overlap_size
        self.dodownsample = dodownsample
        if 0.0 <= highres_rate <= 1.0:
            self.highres_rate = highres_rate
        else:
            raise ValueError("highres_rate should be between 0.0 and 1.0")
        # if edge_size *2 >=frame_size:
        #     raise ValueError("edge_size should be smaller than frame_size/2")
        # self.edge_size = edge_size
        self.visualize = visualize
        self.use8bit = use8bit
        self.tiny = 10000
        self.Ocbit_threshold = Ocbit_threshold
        pass

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
            x_frames.append(value[i:i + self.frame_size+self.overlap_size])
        # 1.2 DCT
        # apply DCT to each frame
        x_dct_frames = []
        for frame in x_frames:
            if len(frame) < self.frame_size+self.overlap_size:
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

        highres_size = int(self.highres_rate * (self.frame_size+self.overlap_size))
        metadata = {}

        if highres_size != 0:
            highres_img = combined_image[:, :highres_size, :]
            highres_max_values = np.max(highres_img)
            highres_min_values = np.min(highres_img)
            highres_img = highres_img / max(abs(highres_max_values), abs(highres_min_values))
            highres_img = self.__UnuniQuantize(highres_img, ununiQuantizeNum)
            highres_img = (highres_img + 1) / 2

            highres_img = (highres_img * 65535)
            highres_img = highres_img.astype(np.uint16)

            jp2_filename = "{}/dct_frames_left.jp2".format(savedir)
            tile_size = (highres_img.shape[0],  highres_img.shape[1])
            jp2 = glymur.Jp2k(
                jp2_filename,
                data=highres_img,
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
            metadata["highres_max_values"] = highres_max_values
            metadata["highres_min_values"] = highres_min_values

        if highres_size != (self.frame_size+self.overlap_size):
            lowres_img = combined_image[:, highres_size:, :]
            lowres_max_values = np.max(lowres_img)
            lowres_min_values = np.min(lowres_img)
            normal_lowres_img = np.copy(lowres_img)
            for i in range(lowres_img.shape[0]):
                line_max = np.max(lowres_img[i])
                line_min = np.min(lowres_img[i])
                normal_lowres_img[i] = lowres_img[i] / max(abs(line_max), abs(line_min))
                
            lowres_img = lowres_img / max(abs(lowres_max_values), abs(lowres_min_values))
            normal_lowres_img = normal_lowres_img[:,int(normal_lowres_img.shape[1]/2):,:]

            threshold = np.max(normal_lowres_img)*self.Ocbit_threshold
            dif_indices = np.where((normal_lowres_img >= threshold) | (normal_lowres_img <= -threshold))
            bitmap = np.zeros(normal_lowres_img.shape[0])
            for i in range(dif_indices[0].shape[0]):
                bitmap[dif_indices[0][i]] = 1
            bitmap = bitmap.astype(np.bool)
            bitmap = self.__createContinuedBitmap(bitmap)
            print("bitmap shape: ", bitmap.shape)
            print("true: ", bitmap[bitmap == True].shape)
            print("false: ", bitmap[bitmap == False].shape)
            unkey_img = lowres_img[bitmap]
            unkey_img = self.__UnuniQuantize(unkey_img, ununiQuantizeNum)
            unkey_img = (unkey_img + 1) / 2
            unkey_img = (unkey_img * 65535)
            unkey_img = unkey_img.astype(np.uint16)
            jp2_filename = "{}/dct_frames_right.jp2".format(savedir)
            tile_size = (unkey_img.shape[0], unkey_img.shape[1])
            if unkey_img.shape[0] != 0:
                jp2 = glymur.Jp2k(
                    jp2_filename,
                    data=unkey_img,
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
            # reverse bitmap
            bitmap = np.logical_not(bitmap)
            bitmap_new = bitmap
            bitmap_new = bitmap_new.astype(np.bool)
            key_img = lowres_img[bitmap_new]
            unkey_img = self.__UnuniQuantize(unkey_img, ununiQuantizeNum)
            key_img = (key_img + 1) / 2
            key_img = (key_img * 65535)
            key_img = key_img.astype(np.uint16)
            tile_size = (key_img.shape[0], key_img.shape[1])
            jp2_filename = "{}/dct_frames_right_low.jp2".format(savedir)
            if key_img.shape[0] != 0:
                jp2 = glymur.Jp2k(
                    jp2_filename,
                    data=key_img,
                    numres=1,
                    cratios=(self.compression_value*2,),
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
            print("Saved bitmap", bitmap_new.shape)
            bit_arr = bitarray(bitmap_new.tolist())
            self.__saveBitArray(bit_arr, "{}/bitmap.bin".format(savedir))
            metadata["lowres_max_values"] = lowres_max_values
            metadata["lowres_min_values"] = lowres_min_values
            metadata["bitarraysize"] = bitmap_new.shape[0]

        metadata["Downsample"] = 1 if self.dodownsample else 0
        metadata["FrameSize"] = self.frame_size
        metadata["overlapSize"] = self.overlap_size
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

        if "highres_max_values" in metadata:
            max_values = metadata["highres_max_values"]
            min_values = metadata["highres_min_values"]
            jp2k = glymur.Jp2k("{}/dct_frames_left.jp2".format(savedir))
            real_image = jp2k[:]
            real_image = real_image.astype(np.float64)
            real_image = ((real_image) / 65535)

            real_image = real_image * 2 - 1
            real_image = self.__unpackUnuniQuantize(real_image, ununiQuantizeNum)

            real_image = real_image * max(abs(max_values), abs(min_values))
            highres_img = real_image
        if "lowres_max_values" in metadata:
            max_values = metadata["lowres_max_values"]
            min_values = metadata["lowres_min_values"]
            bitarraysize = metadata["bitarraysize"]
            if os.path.exists("{}/dct_frames_right.jp2".format(savedir)):
                jp2k_high = glymur.Jp2k(
                    "{}/dct_frames_right.jp2".format(savedir),
                )
                high_image = jp2k_high[:]
                high_image = high_image.astype(np.float64)
                high_image = ((high_image) / 65535)
                high_image = high_image * 2 - 1
                # dif = ((dif) * self.Ocbit_threshold)
                wide = high_image.shape[1]
            if os.path.exists("{}/dct_frames_right_low.jp2".format(savedir)):
                jp2k_low = glymur.Jp2k(
                    "{}/dct_frames_right_low.jp2".format(savedir),
                )
                low_image = jp2k_low[:]
                low_image = low_image.astype(np.float64)
                low_image = ((low_image) / 65535)
                low_image = low_image * 2 - 1
                wide = low_image.shape[1]

            loaded_bit_arr = bitarray()
            with open("{}/bitmap.bin".format(savedir), "rb") as f:
                loaded_bit_arr.fromfile(f)
            bitmap = np.array(loaded_bit_arr.tolist(), dtype=bool)
            bitmap = bitmap[:bitarraysize]

            final_image = np.zeros((bitmap.shape[0], wide, 3))

            key_image_index = 0
            diff_image_index = 0
            for i in range(bitmap.shape[0]):
                if bitmap[i] == True:
                    final_image[i] = low_image[key_image_index]
                    key_image_index += 1
                if bitmap[i] == False:
                    final_image[i] = high_image[diff_image_index]
                    diff_image_index += 1
            
            final_image = self.__unpackUnuniQuantize(final_image, ununiQuantizeNum)
            final_image = final_image * max(abs(max_values), abs(min_values))

            lowres_img = final_image
        if "highres_max_values" in metadata and "lowres_max_values" in metadata:
            real_image = np.concatenate((highres_img, lowres_img), axis=1)
        elif "highres_max_values" in metadata:
            real_image = highres_img
        else:
            real_image = lowres_img

        return real_image[:, :, 0], real_image[:, :, 1], real_image[:, :, 2], metadata

        # reconstruct the original DCT frames

    def __IDCTProcess(self, real_image, channel_name,overlap_size) -> np.array:
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
        if overlap_size == 0:
            series = overlap_frames.reshape(-1)
            return series
        series = overlap_frames[:, :-overlap_size]
        for i in range(1, series.shape[0]):
            for j in range(overlap_size):
                a = overlap_frames[i - 1, -overlap_size + j]
                b = series[i, j]
                distance = float(abs(a - b))
                series[i, j] = a * (j / overlap_size) + b * (1 - j / overlap_size)


        series = series.reshape(-1)
        return series

    def __calculate_psnr(self, original, compressed):
        x_value = original[:, 0]
        y_value = original[:, 1]
        z_value = original[:, 2]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(original)
        mse = 0
        _, indices = nbrs.kneighbors(compressed)
        distances = np.zeros((len(indices), 1))
        for i in range(len(indices)):
            distances[i] = np.linalg.norm(original[indices[i][0]] - compressed[i])
        mse = np.mean(np.square(distances))
        
        reverse_nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(compressed)
        _, reverse_indices = reverse_nbrs.kneighbors(original)
        reverse_distances = np.zeros((len(reverse_indices), 1))
        for i in range(len(reverse_indices)):
            reverse_distances[i] = np.linalg.norm(compressed[reverse_indices[i][0]] - original[i])
        reverse_mse = np.mean(np.square(reverse_distances))
        mse = (mse + reverse_mse) / 2
        # calculate MSE
        # mse /= len(x_value)+len(y_value)+len(z_value)
        x_range = np.max(x_value) - np.min(x_value)
        y_range = np.max(y_value) - np.min(y_value)
        z_range = np.max(z_value) - np.min(z_value)
        max_range = pow(pow(x_range, 2) + pow(y_range, 2) + pow(z_range, 2), 0.5)

        psnr = 10 * np.log10(pow(max_range, 2) / mse)
        x_range = np.max(x_value) - np.min(x_value)
        y_range = np.max(y_value) - np.min(y_value)
        z_range = np.max(z_value) - np.min(z_value)
        max_range = pow(pow(x_range, 2) + pow(y_range, 2) + pow(z_range, 2), 0.5)

        psnr = 10 * np.log10(pow(max_range, 2) / mse)
        return psnr ,distances

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

    def pc2mp3(self, filename, savedir):

        pcd = o3d.io.read_point_cloud(filename)
        np_pcd = np.asarray(pcd.points)
        x_value = np_pcd[:, 0]
        y_value = np_pcd[:, 1]
        z_value = np_pcd[:, 2]
        # remove this dir
        z_value = np_pcd[:, 2]
        # remove this dir
        if os.path.exists(savedir):
            for file in os.listdir(savedir):
                os.remove("{}/".format(savedir) + file)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if self.dodownsample:
            x_value, y_value, z_value = self.__downsample(x_value, y_value, z_value)
            x_value, y_value, z_value = self.__downsample(x_value, y_value, z_value)

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
        x_readed = self.__IDCTProcess(x_read_image, "x",metadata["overlapSize"])
        y_readed = self.__IDCTProcess(y_read_image, "y",metadata["overlapSize"])
        z_readed = self.__IDCTProcess(z_read_image, "z",metadata["overlapSize"])
        if metadata["Downsample"] == 1:
            x_readed, y_readed, z_readed = self.__upsample(x_readed, y_readed, z_readed)
            x_readed, y_readed, z_readed = self.__upsample(x_readed, y_readed, z_readed)

        pc = np.stack((x_readed, y_readed, z_readed), axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        o3d.io.write_point_cloud("{}/saved_point_cloud.ply".format(savedir), pcd)
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
