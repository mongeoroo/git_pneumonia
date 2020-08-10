import numpy as np
import cv2

def min_max_normalization(data):
    rtn = (data-data.min())/(data.max()-data.min()+1e-7)
    return rtn

label = ['NORMAL', 'PNEUMONIA']