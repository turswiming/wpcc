import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import open3d as o3d
import json
import os
import zipfile
path = "./data_input/0001_point_cloud.ply"
def savemp3(audio, filename):
    audio_bytes = audio.tobytes()
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,  
        frame_rate=88200,
        channels=1  # Mono
    )
    
    audio_segment = audio_segment.set_frame_rate(44100) 

    audio_segment.export(filename, 
                        format="ogg",
                        bitrate="128k"
                        )
                        
    
    print(audio[:10])
    print(audio_segment.get_array_of_samples()[:10])
    audio_x = AudioSegment.from_file(filename,format="ogg")
    np_x = np.array(audio_x.get_array_of_samples())
    print(np_x[:10])

def pc2mp3(filename):
    prefix = filename.split("/")[-1].split("_")[0]
    pcd = o3d.io.read_point_cloud(filename)
    np_pcd = np.asarray(pcd.points)
    x_value = np_pcd[:, 0]
    y_value = np_pcd[:, 1]
    z_value = np_pcd[:, 2]
    x_starter = x_value[0]
    y_starter = y_value[0]
    z_starter = z_value[0]

    print(x_value.shape)
    x_min = np.min(x_value)
    x_max = np.max(x_value)
    y_min = np.min(y_value)
    y_max = np.max(y_value)
    z_min = np.min(z_value)
    z_max = np.max(z_value)
    x_value = x_value/max(abs(x_min), abs(x_max))
    y_value = y_value/max(abs(y_min), abs(y_max))
    z_value = z_value/max(abs(z_min), abs(z_max))
    if not os.path.exists("data_output/{}_save/".format(prefix)):
        os.makedirs("data_output/{}_save/".format(prefix))
    audio_samples_float32_x = np.int16(x_value*32767)
    savemp3(audio_samples_float32_x, "data_output/{}_save/x.ogg".format(prefix))

    audio_samples_float32_y = np.int16(y_value*32767)
    savemp3(audio_samples_float32_y, "data_output/{}_save/y.ogg".format(prefix))

    audio_samples_float32_z = np.int16(z_value*32767)
    savemp3(audio_samples_float32_z, "data_output/{}_save/z.ogg".format(prefix))
    savejson = {}
    savejson["x_min"]=x_min
    savejson["x_max"]=x_max
    savejson["y_min"]=y_min
    savejson["y_max"]=y_max
    savejson["z_min"]=z_min
    savejson["z_max"]=z_max
    savejson["x_starter"]=x_starter
    savejson["y_starter"]=y_starter
    savejson["z_starter"]=z_starter
    with open("data_output/{}_save/json.json".format(prefix), "w") as f:
        json.dump(savejson, f)


    #get file size
    x_size = os.path.getsize("data_output/{}_save/x.ogg".format(prefix))
    y_size = os.path.getsize("data_output/{}_save/y.ogg".format(prefix))
    z_size = os.path.getsize("data_output/{}_save/z.ogg".format(prefix))
    with zipfile.ZipFile(
        "data_output/{}_save/zip.zip".format(prefix), 
        'w',
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9
        ) as zipf:
        zipf.write("data_output/{}_save/x.ogg".format(prefix)
                    , arcname="x.ogg")
        zipf.write("data_output/{}_save/y.ogg".format(prefix)
                    , arcname="y.ogg")
        zipf.write("data_output/{}_save/z.ogg".format(prefix)
                    , arcname="z.ogg")
    
    zip_size = os.path.getsize("data_output/{}_save/zip.zip".format(prefix))
    #use zip to compress three fires 2 times
    original_size = os.path.getsize(path)
    print("Original size: {} bytes".format(original_size))
    print("x.ogg size: {} bytes".format(x_size))
    print("y.ogg size: {} bytes".format(y_size))
    print("z.ogg size: {} bytes".format(z_size))
    print("Totalzip size: {} bytes".format(zip_size))
    print("Compression ratio: {:.6f}".format(original_size/zip_size))

if __name__ == "__main__":
    pc2mp3(path)
    print("Done.")