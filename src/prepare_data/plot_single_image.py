import numpy as np
import h5py
import multi_slice_viewer as msv
import scipy.ndimage as ndimage

if __name__ == "__main__":
    
    # filepath = r'C:\Users\dlon450\Documents\4DFlowNetv2\models\4DFlowNet_20210629-1129'
    # filename = r'quicksave_4DFlowNet'
    filepath = r'data/test_280621'
    filename = r'trainG1HR'
    
    input_filepath = f'{filepath}/{filename}.h5'

    # colname = 'mag_v'
    colname = 'v'
    colname = 'mask'

    idx = 0
    with h5py.File(input_filepath, 'r') as hf:
        img = np.asarray(hf.get(colname)[idx])
        # img = np.asarray(hf.get(colname)[2][0])
        print(img.shape)


    # img = ndimage.zoom(img, 2, order=3)
    print(img.shape)
    msv.multi_slice_viewer(img, slice_axis=0)