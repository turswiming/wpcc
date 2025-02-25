from Compressor.PCcompression import PCcompression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "./data_input/000000.ply"
"""
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
if __name__ == "__main__":
    frame_sizes = [4,8,16,32,64,128, 256]
    compress_ratios = [1,3,5,7,9,10,15,20,30,40,50]
    downsamples = [True]
    # BPP = np.load("BPP.npy")
    # psnrs = np.load("psnrs.npy")

    # plt.scatter(BPP.flatten(), psnrs.flatten())
    # plt.xlabel('Bits Per Pixel (BPP)')
    # plt.ylabel('PSNR')
    # plt.title('BPP vs PSNR')
    # plt.grid(True)
    # plt.show()

    # # # 创建一个二维的 numpy 数组
    # # data = np.random.rand(10, 10)
    # #
    # # 创建网格
    # x = np.arange(BPP.shape[1])
    # y = np.arange(BPP.shape[0])
    # x, y = np.meshgrid(x, y)

    # # 创建一个三维图表
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=60, azim=45)
    # ax.dist = 0

    # # 绘制三维图表
    # ax.plot_surface(x, y, psnrs, cmap='viridis')
    # # 设置 x 轴和 y 轴上的 ticks
    # # 设置 x 轴和 y 轴上的 ticks
    # ax.set_xticks(np.arange(len(compress_ratios)))
    # ax.set_xticklabels(compress_ratios)
    # ax.set_yticks(np.arange(len(thresholds)))
    # ax.set_yticklabels(thresholds)
    # # 显示图表
    # plt.show()
    #
    # plt.imshow(BPP)
    # plt.xticks(np.arange(len(threshold_labels)), labels=threshold_labels)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    # plt.imshow(psnrs)
    # plt.xticks(np.arange(len(threshold_labels)), labels=threshold_labels)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    pcc = PCcompression(32,5,False, True)
    pcc.pc2mp3(path, "./data_output/01_save")


    BPP = np.zeros((len(frame_sizes), len(compress_ratios)))
    psnrs = np.zeros((len(frame_sizes), len(compress_ratios)))
    i = 0
    j = 0
    with open ("csvfile.csv", "w") as f:
        f.write("downsample,frame_size,compress_ratio,ratio,psnr\n")
        for downsample in downsamples:
            for  frame_size in frame_sizes:
                for  compress_ratio in compress_ratios:
                    print("-----------------")  
                    print("frame_size: ", frame_size)
                    print("compress_ratio: ", compress_ratio)
                    pcc = PCcompression(frame_size,compress_ratio,downsample,False)
                    bpp, psnr = pcc.pc2mp3(path,"./data_output/01_save")
                    f.write("{},{},{},{},{}\n".format(downsample,frame_size,compress_ratio,bpp,psnr))

