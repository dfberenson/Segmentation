# -*- coding: utf-8 -*-
"""
Created on Wed Jan 8 2017

@author: dfberenson@gmail.com and xies@stanford.edu

@ToDo: Fix thresholding
"""

from skimage import io, filters, morphology, measure, util
from scipy.ndimage import distance_transform_edt
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

"""
File
"""
filename = r'C:\Users\Skotheim Lab\Desktop\Python Scripts\Segmentation\Test images\test.tif'
excel_filename = r'C:\Users\Skotheim Lab\Desktop\Python Scripts\Segmentation\Test images\Test tracking.xlsx'

filename = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4_MMStack_Pos1_red.tif'
filename_chan2 = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4_MMStack_Pos1_green.tif'

filename = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4_MMStack_Pos2_red.tif'
filename_chan2 = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4_MMStack_Pos2_green.tif'

excel_filename = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4 Manual Cell Tracking.xlsx'
um_per_px = 1


#firstcell = input('First cell in this stack: ')
firstcell = 1008
#lastcell = input('Last cell in this stack: ')
lastcell = 1017

start_time = time.clock()
print('\nWorking...\n')

"""
PARAMETERS
"""
smooth_size = 25 # pixels
min_radius = 20
max_radius = 100
min_obj_size_2D = 500; # min px size for objects in 2D
min_obj_size_3D = 1000;
max_px_error = 10

"""
Image I/O
"""

im_stack = io.imread(filename)
im_stack = util.img_as_float(im_stack)
[T,Y,X] = im_stack.shape
mask_clean_stack = []

im_stack_chan2 = io.imread(filename_chan2)
im_stack_chan2 = util.img_as_float(im_stack_chan2)

"""
Preprocessing to generate clean 2D mask
"""

# Compute global Otsu threshold on the image
global_thresh = filters.threshold_otsu(im_stack)
print('Working....\n')

cell_dict = TrackingDataDictionary(excel_filename, firstcell, lastcell)
intensities_series_dict = {}
intensities_series_chan2_dict = {}
for cell in cell_dict:
    intensities_series_dict[cell] = []
    intensities_series_chan2_dict[cell] = []

for t in range(T):
    image = filters.gaussian(im_stack[t] , sigma = 1)
    image_chan2 = filters.gaussian(im_stack_chan2[t], sigma = 1)
    
    global_thresh_individual = filters.threshold_otsu(image)
    global_thresh_individual = 0.0018
    # Threshold the image based on the calculated thresholds

    mask = np.copy(image)
    mask_clean = np.copy(mask)
    label = np.copy(mask).astype(np.int)

    mask = image > global_thresh_individual
    # Clean the masks be removing "salt and pepper"
    mask_clean = morphology.binary_closing(mask)
    mask_clean = morphology.binary_opening(mask_clean)
    mask_clean_stack.append(mask_clean)
    # Use connected components to find the distinct objects
    label = morphology.label(mask_clean).astype(np.int)

    label = RenameLabel(t , label , cell_dict, max_px_error)
    
    # Measures intensities and area so each cell in cell_dict now points to a list in the format
    # [[frame , [x,y] , [intensity, area, mean]]]
    # Also gets a cellwise list of just the intensities as a timeseries

    for cell in cell_dict:
        cellnum = cell
        this_cell_info = cell_dict[cell]
        first_frame = this_cell_info[0][0] - 1
        last_frame = this_cell_info[-1][0] - 1

        this_cell_label = (label == cellnum).astype(np.int)
        if np.any(this_cell_label):                 #if this cell's image has any nonzero values
            properties = measure.regionprops(this_cell_label , image)[0]
            mean = properties.mean_intensity
            area = properties.area
            intensity = mean * area
            properties_chan2 = measure.regionprops(this_cell_label , image_chan2)[0]
            mean_chan2 = properties_chan2.mean_intensity
            area_chan2 = properties_chan2.area
            intensity_chan2 = mean_chan2 * area_chan2
            this_cell_info[t - first_frame].append(intensity)
            this_cell_info[t - first_frame].append(area)
            this_cell_info[t - first_frame].append(mean)
            this_cell_info[t - first_frame].append(intensity_chan2)
            this_cell_info[t - first_frame].append(area_chan2)
            this_cell_info[t - first_frame].append(mean_chan2)
            intensities_series_dict[cell].append(intensity)
            intensities_series_chan2_dict[cell].append(intensity_chan2)
        elif first_frame <= t <= last_frame:
            this_cell_info[t - first_frame].append(-1)
            this_cell_info[t - first_frame].append(-1)
            this_cell_info[t - first_frame].append(-1)
            this_cell_info[t - first_frame].append(-1)
            this_cell_info[t - first_frame].append(-1)
            this_cell_info[t - first_frame].append(-1)

print('Working.....\n')


"""
Get statistics in 3D
"""
#
#
#columns = ('x','y','I','A')
#for i in range(T):
#    props = measure.regionprops(labels) # 2D regionprops
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
#
#######io.imsave('labels.tif',
#######          np.stack(labels).astype(np.int16))


cell_array_dict = {}
for cell in cell_dict:
    cell_array_dict[cell] = np.asarray(cell_dict[cell])

import csv, json
csv_filename = r'E:\DFB imaging experiments\DFB_170203_HMEC_1G_Fucci_4\DFB_170203_HMEC_1G_Fucci_4 cell_dict.csv'
with open(csv_filename, 'a') as f:
    writer = csv.DictWriter(f, fieldnames = ['cellnum' , 'celldata'])
    writer.writeheader()
    for cell in cell_array_dict:
        writer.writerow({'cellnum' : cell , 'celldata' : json.dumps(cell_array_dict[cell].tolist())})
#Don't ask me why I need to convert it to a numpy array and then back to a list to get json.dumps() to work

#for cell in intensities_series_dict:
#    plt.plot(intensities_series_dict[cell])
#    plt.plot(intensities_series_chan2_dict[cell])

print ('\n\nTime elapsed (s): '),
print (int(time.clock() - start_time))