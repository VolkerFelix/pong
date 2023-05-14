import numpy as np
import cv2

def store_step(f_rgb_array_buffer, f_env):
    rgb_array = f_env.render()
    f_rgb_array_buffer.append(rgb_array.astype(np.uint8))

def convert_and_release_video(f_rgb_array_buffer, f_frame_dim):
    out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (f_frame_dim['hight'], f_frame_dim['width']))
    for rgb_array in f_rgb_array_buffer:
        out_video.write(rgb_array)

    out_video.release()
    print("Released new video!")

