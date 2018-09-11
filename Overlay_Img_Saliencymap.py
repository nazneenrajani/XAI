from skimage import transform, filters
import numpy as np
from PIL import Image
from numpy import array
from scipy.misc import imsave
import matplotlib.pyplot as plt
import sys

def showAttMap(img, attMap, qid, att,thresh,blur = True, overlap = True):
    a = np.array(attMap)
    a[:,:,2] *=0
    a = Image.fromarray(a)
    img =img.resize((attMap.size),Image.ANTIALIAS)
    new_img = Image.blend(img, a, 0.4)
    imsave('thresh/'+att+qid+"-"+thresh+'.png', new_img)

f = open("testdev_sys3_qid","r")
qid = [line.rstrip() for line in f.readlines()]
thresh = ['0.1','0.2','0.3','0.4']
att = ['l-', 'h-','r-','t-a-']
for q in qid:
	im = q[:-1]
	print(im)
	a = Image.open("../VQA/test2015/COCO_test2015_"+im.zfill(12)+".jpg")
	for at in att:
		for t in thresh:
			b = Image.open("thresh/"+at+q+'-'+t+".png")
			showAttMap(a,b,q,at,t)
