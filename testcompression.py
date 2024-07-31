from Compressor.PCcompression import PCcompression
import numpy as np

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
    pcc = PCcompression(4, 1, False, True, False)
    pcc.pc2mp3(path, "./data_output/01_save")

    pcc.compress(path, "./data_output/01_save")
    pcc2 = PCcompression("./data_output/01_save")
    pcc2.decompress("./data_output/01_save/01.ply")

    frame_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    cp_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    ratios = np.zeros((len(frame_sizes), len(cp_levels)))
    psnrs = np.zeros((len(frame_sizes), len(cp_levels)))
    i = 0
    j = 0
    with open("csvfile.csv", "w") as f:
        f.write("frame_size,cp_level,ratio,psnr\n")
        for frame_size in frame_sizes:
            for cplevel in cp_levels:
                print("-----------------")
                print("frame_size: ", frame_size)
                print("cplevel: ", cplevel)
                pcc = PCcompression(frame_size, cplevel, True, False, False)
                ratios[i, j], psnrs[i, j] = pcc.pc2mp3(path, "./data_output/01_save")
                f.write("{},{},{},{}\n".format(frame_size, cplevel, ratios[i, j], psnrs[i, j]))
                print("-----------------")
                j += 1
            j = 0
            i += 1
    # np.save("ratios.npy", ratios)
    # np.save("psnrs.npy", psnrs)
    # plt.imshow(psnrs)
    # plt.xticks(np.arange(len(cp_levels)), labels=cp_levels)
    # plt.yticks(np.arange(len(frame_sizes)), labels=frame_sizes)
    # plt.show()
    # plt.close()
    print("Done.")
