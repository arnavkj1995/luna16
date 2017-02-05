import numpy as np
import SimpleITK as sitk
import scipy.misc

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = (sitk.GetArrayFromImage(itkimage))
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def save_itk(im, filename):
    sitk.WriteImage(sitk.GetImageFromArray(im), filename)
    
if __name__ == "__main__":
    image, origin, spacing = load_itk('1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd')
    shapes = [slice.shape for slice in image]
    scipy.misc.imshow(image[70,:,:])
    print shapes
    print 'Shape:', image.shape
    print 'Origin:', origin
    print 'Spacing:', spacing
        