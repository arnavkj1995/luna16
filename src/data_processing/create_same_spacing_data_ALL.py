import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage
import pandas as pd
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
import time

from xyz_utils import load_itk, world_2_voxel, voxel_2_world, save_itk
from joblib import Parallel, delayed

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi

RESIZE_SPACING = [1, 1, 1]
SAVE_FOLDER_image = '1_1_1mm_slices_lung_ALL'
SAVE_FOLDER_lung_mask = '1_1_1mm_slices_lung_masks_ALL'
SAVE_FOLDER_nodule_mask = '1_1_1mm_slices_nodule_ALL'


def seq(start, stop, step=1):
	n = int(round((stop - start)/float(step)))
	if n > 1:
		return([start + step*i for i in range(n+1)])
	else:
		return([])

def draw_circles(image,cands,origin,spacing):

	#make empty matrix, which will be filled with the mask
	image_mask = np.zeros(image.shape)

	#run over all the nodules in the lungs
	for ca in cands.values:

		#get middel x-,y-, and z-worldcoordinate of the nodule
		radius = np.ceil(ca[4])/2
		coord_x = ca[1]
		coord_y = ca[2]
		coord_z = ca[3]
		image_coord = np.array((coord_z,coord_y,coord_x))

		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(image_coord,origin,spacing)

		#determine the range of the nodule
		noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

		#create the mask
		for x in noduleRange:
			for y in noduleRange:
				for z in noduleRange:
					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
					if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
						image_mask[np.round(coords[0]),np.round(coords[1]),np.round(coords[2])] = int(1)
	
	return image_mask



def get_segmented_lungs(im, plot=False):
	
	'''
	This funtion segments the lungs from the given 2D slice.
	'''
	if plot == True:
		f, plots = plt.subplots(8, 1, figsize=(5, 40))
	'''
	Step 1: Convert into a binary image. 
	'''
	binary = im < 604
	if plot == True:
		plots[0].axis('off')
		plots[0].imshow(binary, cmap=plt.cm.bone) 
	'''
	Step 2: Remove the blobs connected to the border of the image.
	'''
	cleared = clear_border(binary)
	if plot == True:
		plots[1].axis('off')
		plots[1].imshow(cleared, cmap=plt.cm.bone) 
	'''
	Step 3: Label the image.
	'''
	label_image = label(cleared)
	if plot == True:
		plots[2].axis('off')
		plots[2].imshow(label_image, cmap=plt.cm.bone) 
	'''
	Step 4: Keep the labels with 2 largest areas.
	'''
	areas = [r.area for r in regionprops(label_image)]
	areas.sort()
	if len(areas) > 2:
		for region in regionprops(label_image):
			if region.area < areas[-2]:
				for coordinates in region.coords:                
					   label_image[coordinates[0], coordinates[1]] = 0
	binary = label_image > 0
	if plot == True:
		plots[3].axis('off')
		plots[3].imshow(binary, cmap=plt.cm.bone) 
	'''
	Step 5: Erosion operation with a disk of radius 2. This operation is 
	seperate the lung nodules attached to the blood vessels.
	'''
	selem = disk(2)
	binary = binary_erosion(binary, selem)
	if plot == True:
		plots[4].axis('off')
		plots[4].imshow(binary, cmap=plt.cm.bone) 
	'''
	Step 6: Closure operation with a disk of radius 10. This operation is 
	to keep nodules attached to the lung wall.
	'''
	selem = disk(10)
	binary = binary_closing(binary, selem)
	if plot == True:
		plots[5].axis('off')
		plots[5].imshow(binary, cmap=plt.cm.bone) 
	'''
	Step 7: Fill in the small holes inside the binary mask of lungs.
	'''
	edges = roberts(binary)
	binary = ndi.binary_fill_holes(edges)
	if plot == True:
		plots[6].axis('off')
		plots[6].imshow(binary, cmap=plt.cm.bone) 
	'''
	Step 8: Superimpose the binary mask on the input image.
	'''
	# get_high_vals = binary == 0
	# im[get_high_vals] = 0
	# if plot == True:
	#     plots[7].axis('off')
	#     plots[7].imshow(im, cmap=plt.cm.bone) 
		
	return binary

def segment_lung_from_ct_scan(ct_scan):
	seg_ct_scan = np.zeros(ct_scan.shape)
	for i in range(0, ct_scan.shape[2]):
		seg_ct_scan[:,:,i] = get_segmented_lungs(ct_scan[:,:,i])
	return seg_ct_scan

def create_slices(imagePath, maskPath, cads):
	#if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
	img, origin, spacing = load_itk(imagePath)
	img_num = 79

	#determine the cads in a lung from csv file
	imageName = imagePath.replace('.mhd','')
	image_cads = cads[cads['seriesuid'] == imageName]

	#calculate resize factor
	resize_factor = spacing / RESIZE_SPACING
	new_real_shape = img.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize = new_shape / img.shape
	new_spacing = spacing / real_resize
	
	#resize image & resize lung-mask
	lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
	print lung_img.shape
	lung_img = lung_img + 1024
	lung_mask = segment_lung_from_ct_scan(lung_img)
	lung_img = lung_img - 1024

	#create nodule mask
	nodule_mask = draw_circles(lung_img,image_cads,origin,new_spacing)

	lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros((lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

	#save slices
	original_shape = lung_img.shape
	
	for z in range(lung_img.shape[0]):
		offset = (512 - original_shape[1])
		upper_offset = np.round(offset/2)
		lower_offset = offset - upper_offset

		new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

		lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
		lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
		nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

	save_itk(lung_img_512, imageName + '_lung_img.mhd')
	save_itk(lung_mask_512, imageName + '_lung_mask.mhd')
	save_itk(nodule_mask_512, imageName + '_nodule_mask.mhd')
	
	# for img_num in range(40, 300, 40):
	# 	scipy.misc.imshow(lung_img_512[img_num, :, :])
	# 	scipy.misc.imshow(lung_mask_512[img_num, :, :])
	# 	scipy.misc.imshow(nodule_mask_512[img_num, :, :])


def createImageList(subset, cads):
	imagesWithNodules = []
	subsetDir = '../../../data/luna16/original_lungs/subset_{}'.format(subset)
	imagePaths = glob.glob("{}/*.mhd".format(subsetDir))
	return imagePaths

if __name__ == "__main__":
	cads = pd.read_csv("/home/arnav/CSVFILES/annotations.csv")
	# for subset in range(10):
	subset = 9
	start_time = time.time()
	print '{} - Processing subset'.format(time.strftime("%H:%M:%S")), subset
	imagePaths = '1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd'
	create_slices(imagePaths, imagePaths, cads)
	#Parallel(n_jobs=24, verbose=5)(delayed(create_slices)(imagePath,imagePath.replace('../../../data/luna16/original_lungs/subset_{}'.format(subset), 'original_lung_masks'), cads) for imagePath in imagePaths)
	print '{} - Processing subset {} took {} seconds'.format(time.strftime("%H:%M:%S"),subset,np.floor(time.time()-start_time))
	print