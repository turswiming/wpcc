# wpcc: wave point cloud compression
This project leverage the potential of music and wave to store point cloud in a high compression rate and high psnr.  
![Compression Ratio vs. PSNR](./CRvsPSNR.png)  





![point cloud compression level 1](./PCClevel1.png)
point cloud comresssion level 1  
compression ratio 4.3

![point cloud compression level 5](./PCClevel5.png)
point cloud comresssion level 5  
compression ratio 9.98


## how to init this project
``` bash  
pip install -r requirements.txt
```
## usage

1. open testcompression.py
1. use this sample code 
    ```python
    pcc = PCcompression(box_size, frame_size, compression_value, visualize, use8bit)
    pcc.pc2mp3(path)
    ```
1. and the compression ratio and PSNR will be printed.
