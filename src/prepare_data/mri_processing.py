import numpy as np
import h5py
import multi_slice_viewer as msv
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom
from h5functions import save_to_h5

def load_mhd(filename):
    '''
        https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    img = sitk.GetArrayFromImage(itkimage)
    return img

if __name__ == '__main__':
    result = True
    if result:
        # fn = "data/real_mri_data/Cardiohance034_aorta.mha"
        infn = "result/real_mri_data/csp_mri.h5"
        # ofn = "result/real_mri_data/csp_mri_vtk.h5"
        # mask = load_mhd(fn)
        # mask4x = zoom(mask, (4, 4, 4))

        # mask4x = np.transpose(mask4x, (0, 2, 1))
        with h5py.File(infn, 'r') as hf:
            u = np.asarray(hf.get('u')[5,:,:,:])
            v = np.asarray(hf.get('v')[5,:,:,:])
            w = np.asarray(hf.get('w')[5,:,:,:])
        
        img = np.sqrt(u**2 + v**2 + w**2)
        # img = img * mask

        msv.multi_slice_viewer(img, slice_axis=0,  clim=[0., 2.], color='inferno')
        # save_to_h5(ofn, "u", u*mask4x)
        # save_to_h5(ofn, "v", v*mask4x)
        # save_to_h5(ofn, "w", w*mask4x)
    else:
        fn = "data/real_mri_data/Cardiohance034_aorta.mha"
        mask = load_mhd(fn)
        infn = "data/real_mri_data/ch34_mri.h5"

        mask = np.transpose(mask, (0, 2, 1))
        with h5py.File(infn, 'r') as hf:
            u = np.asarray(hf.get('u')[5,:,:,:])
            v = np.asarray(hf.get('v')[5,:,:,:])
            w = np.asarray(hf.get('w')[5,:,:,:])

        img = np.sqrt(u**2 + v**2 + w**2)
        msv.multi_slice_viewer(img*mask, slice_axis=0, color='inferno')
        # save_to_h5(infn, "u_masked", u*mask)
        # save_to_h5(infn, "v_masked", v*mask)
        # save_to_h5(infn, "w_masked", w*mask)