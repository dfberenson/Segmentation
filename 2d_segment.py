# -*- coding: utf-8 -*-
"""
Created on Wed Jan 8 2017

@author: dfberenson@gmail.com and xies@stanford.edu
"""

import os
from skimage import io, filters, morphology, measure, util
from scipy.ndimage import distance_transform_edt
import pandas as pd

"""
File
"""

filename = r'C:\Users\Skotheim Lab\Desktop\Python Scripts\Segmentation\Test images\test.tif'
excel_filename = r'C:\Users\Skotheim Lab\Desktop\Python Scripts\Segmentation\Test images\Test tracking.xlsx'
um_per_px = 1

"""
PARAMETERS
"""
smooth_size = 25 # pixels
min_radius = 20
max_radius = 100

min_obj_size_2D = 500; # min px size for objects in 2D
min_obj_size_3D = 1000;

"""
Image I/O
"""

im_stack = io.imread(filename)
im_stack = util.img_as_float(im_stack)
[Y,X,T] = im_stack.shape

"""
Preprocessing to generate clean 2D mask

"""

# Compute global Otsu threshold on the image
global_thresh = filters.threshold_otsu(im_stack)

global_thresh_individual = range(T)
for t in range(T):
    global_thresh_individual[t] = filters.threshold_otsu(im_stack[:,:,t])

# Threshold the image based on the calculated thresholds

mask = np.copy(im_stack)
mask_clean = np.copy(mask)
labels = np.copy(mask).astype(np.int)
for t in range(T):
    mask[:,:,t] = im_stack[:,:,t] > global_thresh_individual[t]
    # Clean the masks be removing "salt and pepper"
    mask_clean[:,:,t] = morphology.binary_closing(mask[:,:,t])
    mask_clean[:,:,t] = morphology.binary_opening(mask_clean[:,:,t])
    # Use connected components to find the distinct objects
    labels[:,:,t] = morphology.label(mask_clean[:,:,t]).astype(np.int)


cell_dict = TrackingDataDictionary(excel_filename)
cell_names = XlsxSheetNames(excel_filename)

labels = RenameLabels(labels,cell_dict)
        
    

"""
Get statistics in 3D
"""


columns = ('x','y','I','A')
for i in range(T):
    props = measure.regionprops(labels) # 2D regionprops
    
                           

for z, frame in enumerate(labels):
    f_prop = measure.regionprops(frame.astype(np.int),
                intensity_image=dapi[z,:,:])
    for d in f_prop:
        radius = (d.area / np.pi)**0.5
        if (min_radius < radius < max_radius):
            properties.append([d.weighted_centroid[0],
                              d.weighted_centroid[1],
                              z, d.mean_intensity * d.area,
                              radius])
            indices.append(d.label)

if not len(indices):
    all_props = pd.DataFrame([],index=[])
indices = pd.Index(indices, name='label')
properties = pd.DataFrame(properties, index=indices, columns=columns)
properties['I'] /= properties['I'].max()

#"""
#Watershed
#1) Get distance transform (Euclidean)
#2) Generate foreground markers (object markers) from the local maxima of dist
#transform; markers no closer than specified threshold (typically 10 px). Dilate
#markers for easy visualization
#4) Mark background as 0
#5) Perform watershed
#
#"""
#
## get Distance transform of cleaned up mask (euclidean)
#distTransform = distance_transform_edt(mask_clean)
#watershedImg = -distTransform / distTransform.max() + 1
#
### Get local maxima from bwdist image and filter 
##I = filters.gaussian(distTransform,4)
##I = feature.peak_local_max(I, min_distance=20,indices=False)
##for i in range(5):
##    I = morphology.dilation(I)
#
#markers = measure.label(sure_fg)
##markers += 1
##markers[sure_bg] = 0
#
## Perform watershed
#labels = morphology.watershed( watershedImg, markers)
#objectIDs = np.setdiff1d( np.unique(labels), [0,1] )

#"""
#Get statistics in 3D
#"""
#
#properties = measure.regionprops(labels, dapi) # 3D regionprops (some properties are not supported yet)
#columns = ('x','y','z','I','w')
#
#
#
#for z, frame in enumerate(labels):
#    f_prop = measure.regionprops(frame.astype(np.int),
#                intensity_image=dapi[z,:,:])
#    for d in f_prop:
#        radius = (d.area / np.pi)**0.5
#        if (min_radius < radius < max_radius):
#            properties.append([d.weighted_centroid[0],
#                              d.weighted_centroid[1],
#                              z, d.mean_intensity * d.area,
#                              radius])
#            indices.append(d.label)
#
#if not len(indices):
#    all_props = pd.DataFrame([],index=[])
#indices = pd.Index(indices, name='label')
#properties = pd.DataFrame(properties, index=indices, columns=columns)
#properties['I'] /= properties['I'].max()
#
#"""
#Orthogonal projection
#"""
#plot_stack_projections(labels)

"""
Preview
u
"""
#
#print "Total # of final labels: %d " % (objectIDs.size)

io.imsave('labels.tif',
          np.stack(labels).astype(np.int16))
