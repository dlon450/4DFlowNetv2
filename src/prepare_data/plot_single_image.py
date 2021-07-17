import numpy as np
import h5py
import multi_slice_viewer as msv
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    quicksave = True

    if quicksave:
        upsample_rate = 4
        filepath = r'C:\Users\dlon450\Documents\4DFlowNetv2\models\4DFlowNet_20210705-1126'
        filename = r'quicksave_4DFlowNet'
    else:
        filepath = r'data/combined'
        filename = r'trainG5HR'
    
    input_filepath = f'{filepath}/{filename}.h5'

    # colname = 'mag_v'
    for colname in ['v']:
        idx = 0
        for i in range(17,18):
            with h5py.File(input_filepath, 'r') as hf:
                if quicksave:
                    img = np.asarray(hf.get(colname)[i][idx])
                    img2 = np.asarray(hf.get('hr_v')[0])
                    img3 = np.asarray(hf.get('lr_v')[0])
                    img3 = np.squeeze(img3)
                    print(img3.shape)
                else:
                    img = np.asarray(hf.get(colname)[idx])

            # img = ndimage.zoom(img, 2, order=3)
            # msv.multi_slice_viewer(img, slice_axis=0)
            minval, maxval = np.min(img),np.max(img)
            plt.subplot(131), plt.imshow(img[24], cmap='jet', clim=[minval, maxval])
            plt.subplot(132), plt.imshow(img2[24], cmap='jet', clim=[minval, maxval])
            plt.subplot(133), plt.imshow(img3[24//upsample_rate], cmap='jet', clim=[minval, maxval])
            plt.colorbar()
            plt.show()      
    
    print(img.shape)