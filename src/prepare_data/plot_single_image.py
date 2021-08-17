import numpy as np
import h5py
import multi_slice_viewer as msv
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    quicksave = False

    if quicksave:
        upsample_rate = 4
        filepath = r'C:\Users\longd\OneDrive - The University of Auckland\Documents\2021\ENGSCI 700A\4DFlowNetv2\models\4DFlowNet'
        filename = r'quicksave_4DFlowNet'
    else:
        filepath = r'models'
        filename = r'resnet_noAliasing_G4'
        mask = False
        maskname = 'mask'


    input_filepath = f'{filepath}\{filename}.h5'

    # colname = 'mag_v'
    for colname in ['u']:
        idx = 7
        for i in range(10,61,25):
            with h5py.File(input_filepath, 'r') as hf:
                if quicksave:
                    img = np.asarray(hf.get(colname)[25][idx])
                    img2 = np.asarray(hf.get('hr_v')[idx])
                    img3 = np.asarray(hf.get('lr_v')[idx])
                    img3 = np.squeeze(img3)
                    print(img3.shape)
                else:
                    if mask:
                        img = np.asarray(hf.get(maskname)[0])
                    else:
                        img = np.asarray(hf.get(colname)[i])

            if quicksave:
                minval, maxval = np.min(img),np.max(img)
                # minval, maxval = -0.1, 0.2
                plt.subplot(131), plt.imshow(img[24], cmap='jet', clim=[minval, maxval])
                plt.subplot(132), plt.imshow(img2[24], cmap='jet', clim=[minval, maxval])
                plt.subplot(133), plt.imshow(img3[24//upsample_rate], cmap='jet', clim=[minval, maxval])
                plt.colorbar()
                plt.show()    
            else:
                minval = -0.10
                maxval = .7
                msv.multi_slice_viewer(img, slice_axis=0, clim=[minval, maxval])
                # msv.multi_slice_viewer(img, slice_axis=0)
            
    print(img.shape)