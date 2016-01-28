import os
import skimage
import sklearn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from sklearn.feature_extraction import image
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

from time import time



filename = 'texture4.jpg'
camera = io.imread(filename,as_grey=True).astype(np.float32)
x=camera.shape[0]
y=camera.shape[1]


plt.figure() 
plt.imshow(camera) 

n_colors=4


t0 = time()
#camera=camera.reshape(1,camera.shape[0]*camera.shape[1])
sample = shuffle(camera, random_state=0)[:5000]
kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0,n_init=200,).fit(sample)
print("done in %0.3fs." % (time() - t0))

t0 = time()
X = camera.reshape((-1, 1))
labels = kmeans.fit(X)
print("done in %0.3fs." % (time() - t0))
#

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_
camerapic = np.choose(labels, values)
camerapic.shape = camera.shape
#kernel = np.ones((5,5),np.uint8)
#camerapic = cv2.morphologyEx(camerapic, cv2.MORPH_CLOSE, kernel)

plt.figure() 
plt.imshow(camerapic,interpolation='none')

#
#segments_fz = felzenszwalb(camera, scale=1000, sigma=0.5, min_size=5)
#plt.figure() 
#plt.imshow(segments_fz,interpolation='none')