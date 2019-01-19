import numpy as np
from PIL import Image
import scipy.io as sio
import scipy.ndimage

img_info = sio.loadmat("./data/mall/mall_gt.mat")

def get_head_count(frame_num: int):
    return img_info["count"][frame_num][0]

def get_head_locations(frame_num: int):
    return img_info["frame"][0][frame_num][0][0][0].round().astype(int)

def frame_to_array(frame_num: int):
    img = Image.open("./data/mall/frames/seq_{}.jpg".format(str(frame_num+1).zfill(6)))
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = data / 255
    return data

def loc_to_img_arr(loc_arr, img_arr, scale_down=1):
    base_arr = np.zeros(shape=(img_arr.shape[0]//scale_down, img_arr.shape[1]//scale_down))
    for pos in loc_arr:
        base_arr[pos[1]//scale_down][pos[0]//scale_down] = 1
    base_arr = np.expand_dims(base_arr, axis=-1)
    return base_arr

def blur_img(img_arr, blur_rad):
    new_den = scipy.ndimage.filters.gaussian_filter(img_arr, blur_rad, mode='constant')
    return new_den

def get_full_frame(frame_num: int, scale_down=4, blur=False, blur_rad=3):
    frame_arr = frame_to_array(frame_num)
    loc_arr = get_head_locations(frame_num)
    loc_img_arr = loc_to_img_arr(loc_arr, frame_arr, scale_down=scale_down)
    if blur:
        loc_img_arr_den = blur_img(loc_img_arr, blur_rad)
        loc_img_arr = loc_img_arr_den*(np.sum(loc_img_arr) / np.sum(loc_img_arr_den))
    return {
        "locations": loc_arr,
        "loc_img": loc_img_arr.reshape(loc_img_arr.shape[0], loc_img_arr.shape[1], 1),
        "image": frame_arr
    }

def get_frames(n_from: int, n_to: int):
    """Returns two arrays, one is the actual images and the other is the true density maps of people
    
    n_from is the lower inclusive bound of the frame number
    n_to is the upper exclusive bound of the frame number
    """
    size = n_to-n_from
    img_arr = np.zeros((size, 480, 640, 3))
    density_arr = np.zeros((size, 480, 640, 1))
    for i in range(size):
        frame_num = n_from+i
        frame = get_full_frame(frame_num)
        
        img_arr[i] = frame["image"]
        density_arr[i] = frame["loc_img"]
    return img_arr, density_arr