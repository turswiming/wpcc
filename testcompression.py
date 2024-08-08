from Compressor.PCcompression import PCcompression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "./data_input/000000.ply"
"""
usage:

    pcc = PCcompression(16, 1, dodownsample=True, visualize=True, use8bit=False)
    pcc.compress("./path/to/point/cloud", "./data_output/01_save")
    pcc.pc2mp3("./path/to/compressed_directory")
    #and the compression ratio and PSNR will be printed.

    pcc2 = PCcompression("./data_output/01_save")
    pcc2.decompress("./data_output/01_save/01.ply")
    
"""
if __name__ == "__main__":
    frame_sizes = [4,8,16,32,64,128, 256, 512,1024,2048,4096,8192]
    hires_rates = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    thresholds = [1/255,2/255,3/255,4/255,5/255,6/255,7/255,8/255,9/255,10/255]
    threshold_labels = ["1/255","2/255","3/255","4/255","5/255","6/255","7/255","8/255","9/255","10/255"]
    
    # ratios = np.load("ratios.npy")
    # psnrs = np.load("psnrs.npy")
    # # # 创建一个二维的 numpy 数组
    # # data = np.random.rand(10, 10)
    # #
    # # 创建网格
    # x = np.arange(psnrs.shape[1])
    # y = np.arange(psnrs.shape[0])
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
    # ax.set_xticks(np.arange(len(threshold_labels)))
    # ax.set_xticklabels(threshold_labels)
    # ax.set_yticks(np.arange(len(hires_rates)))
    # ax.set_yticklabels(hires_rates)
    # # 显示图表
    # plt.show()
    #
    # plt.imshow(ratios)
    # plt.xticks(np.arange(len(threshold_labels)), labels=threshold_labels)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    # plt.imshow(psnrs)
    # plt.xticks(np.arange(len(threshold_labels)), labels=threshold_labels)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    pcc = PCcompression(16,10,0,5/10,2,True, True, False)
    pcc.pc2mp3(path, "./data_output/01_save")


    ratios = np.zeros((len(frame_sizes), len(thresholds)))
    psnrs = np.zeros((len(frame_sizes), len(thresholds)))
    i = 0
    j = 0
    with open ("csvfile.csv", "w") as f:
        f.write("hires_rate,threshold,ratio,psnr\n")
        for  hires_rate in hires_rates:
            for  threshold in thresholds:
                print("-----------------")  
                print("frame_size: ", hires_rate)
                print("cplevel: ", threshold)
                pcc = PCcompression(32,10,0,hires_rate,2,True,False,False)
                ratios[i,j], psnrs[i,j] = pcc.pc2mp3(path,"./data_output/01_save")
                f.write("{},{},{},{}\n".format(hires_rate,threshold,ratios[i,j],psnrs[i,j]))
                j+=1
            j = 0
            i+=1
    np.save("ratios.npy", ratios)
    np.save("psnrs.npy", psnrs)
    plt.imshow(ratios)
    plt.xticks(np.arange(len(hires_rates)), labels=hires_rates)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    plt.show()
    plt.close()
    plt.imshow(psnrs)
    plt.xticks(np.arange(len(hires_rates)), labels=hires_rates)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    plt.show()
    plt.close()
    print("Done.")
