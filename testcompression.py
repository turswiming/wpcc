
from Compressor.PCcompression import PCcompression
import numpy as np
import matplotlib.pyplot as plt
path = "./data_input/000000.ply"

if __name__ == "__main__":
    frame_sizes = [4,8,16,32,64,128, 256, 512,1024,2048,4096,8192]
    highress = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # ratios = np.load("ratios.npy")
    # psnrs = np.load("psnrs.npy")
    # plt.imshow(ratios)
    # plt.xticks(np.arange(len(highress)), labels=highress)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    # plt.imshow(psnrs)
    # plt.xticks(np.arange(len(highress)), labels=highress)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    pcc = PCcompression(16,1,1,True, True, False)
    pcc.pc2mp3(path, "./data_output/01_save")


    ratios = np.zeros((len(frame_sizes), len(highress)))
    psnrs = np.zeros((len(frame_sizes), len(highress)))
    i = 0
    j = 0
    with open ("csvfile.csv", "w") as f:
        f.write("frame_size,cp_level,ratio,psnr\n")
        for  frame_size in frame_sizes:
            for  highres in highress:
                print("-----------------")  
                print("frame_size: ", frame_size)
                print("cplevel: ", highres)
                pcc = PCcompression(frame_size,1,highres,True,False,False)
                ratios[i,j], psnrs[i,j] = pcc.pc2mp3(path,"./data_output/01_save")
                f.write("{},{},{},{}\n".format(frame_size,highres,ratios[i,j],psnrs[i,j]))
                j+=1
            j = 0
            i+=1
    np.save("ratios.npy", ratios)
    np.save("psnrs.npy", psnrs)
    plt.imshow(ratios)
    plt.xticks(np.arange(len(highress)), labels=highress)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    plt.show()
    plt.close()
    plt.imshow(psnrs)
    plt.xticks(np.arange(len(highres)), labels=highres)
    plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    plt.show()
    plt.close()
    print("Done.")