import numpy as np
from PIL import Image 

def bayer_to_rgb(bayer_file):
    bayer = np.fromfile(open(bayer_file, 'rb'), dtype=np.uint8).reshape(1080, 1920)
    

    R = (bayer[0::2, 0::2]) 
    R = R.astype(np.float32)
    R = R / 255  
    R = R * 3
    R = R.clip(0,1)
    R = R * 255 
    R = R.astype(np.uint8)

    G = (bayer[0::2, 1::2]) 
    G = G.astype(np.float32)
    G = G / 255  
    G = G * 2
    G = G.clip(0,1)
    G = G * 255 
    G = G.astype(np.uint8)

    B = (bayer[1::2, 1::2]) 
    B = B.astype(np.float32)
    B = B / 255  
    B = B * 3
    B = B.clip(0,1)
    B = B * 255
    B = B.astype(np.uint8)

    
    bayer = np.dstack((R,G,B))
    bayer = bayer.clip(0,255)
    img = Image.fromarray(bayer)

    return img