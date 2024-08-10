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
    frame_sizes = [4,8,16,32,64,128, 256, 512,1024,2048,4096,8192]
    hires_rates = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    compress_ratios = [1,2,3,4,5,6,7,8,9,10]
    # BPP = np.load("BPP.npy")
    # psnrs = np.load("psnrs.npy")
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
    pcc = PCcompression(64,1,1,0/10,0,True, True, False)
    pcc.pc2mp3(path, "./data_output/01_save")


    BPP = np.zeros((len(thresholds), len(compress_ratios)))
    psnrs = np.zeros((len(thresholds), len(compress_ratios)))
    i = 0
    j = 0
    with open ("csvfile.csv", "w") as f:
        f.write("threshold,compress_ratio,ratio,psnr\n")
        for  threshold in thresholds:
            for  compress_ratio in compress_ratios:
                print("-----------------")  
                print("threshold: ", threshold)
                print("compress_ratio: ", compress_ratio)
                pcc = PCcompression(32,compress_ratio,0,threshold,2,True,False,False)
                BPP[i,j], psnrs[i,j] = pcc.pc2mp3(path,"./data_output/01_save")
                f.write("{},{},{},{}\n".format(threshold,compress_ratio,BPP[i,j],psnrs[i,j]))
                j+=1
            j = 0
            i+=1
    np.save("BPP.npy", BPP)
    np.save("psnrs.npy", psnrs)
    plt.imshow(BPP)
    plt.xticks(np.arange(len(thresholds)), labels=thresholds)
    plt.yticks(np.arange(len(compress_ratios)), labels=compress_ratios)
    plt.show()
    plt.close()
    plt.imshow(psnrs)
    plt.xticks(np.arange(len(thresholds)), labels=thresholds)
    plt.yticks(np.arange(len(compress_ratios)), labels=compress_ratios)
    plt.show()
    plt.close()
    print("Done.")
