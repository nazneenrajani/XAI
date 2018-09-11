from skimage import transform, filters
import numpy as np
from PIL import Image
from numpy import array
from scipy.misc import imsave
import matplotlib.pyplot as plt
import sys

def showAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    print(img.shape[:2])
    attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'edge')
    print(attMap.shape)
    attMap = attMap[:,:,0]
    if blur:
        attMap = filters.gaussian(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 2, 2)
    if overlap:
        attMap = 1*(1-attMap).reshape(attMap.shape + (1,))*img + (attMap).reshape(attMap.shape + (1,)) * attMapV #(1-attMap**0.7).reshape(attMap.shape) #1*(1-attMap**0.5)*img + ((attMap**0.5)*attMap)
    imsave('output/'+sys.argv[3], attMap)


a = np.asarray(Image.open(sys.argv[1]))
b = np.asarray(Image.open(sys.argv[2]))
a.setflags(write=1)
b.setflags(write=1)
showAttMap(a,b)
