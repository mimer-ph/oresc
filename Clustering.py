# KMeans clustering
#


#import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from time import time


#importing the texture
filename = 'texture4.jpg'
camera = io.imread(filename,as_grey=True).astype(np.float32)
x=camera.shape[0]
y=camera.shape[1]

# displaying it
plt.figure() 
plt.imshow(camera) 

# Setting the clustering settings
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

# Display the labels
values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_
camerapic = np.choose(labels, values)
camerapic.shape = camera.shape

# not needed : morphological closure
#kernel = np.ones((5,5),np.uint8)
#camerapic = cv2.morphologyEx(camerapic, cv2.MORPH_CLOSE, kernel)

plt.figure() 
plt.imshow(camerapic,interpolation='none')
# alt: matshow