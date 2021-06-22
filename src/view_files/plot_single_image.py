import numpy as np
import h5py
import multi_slice_viewer as msv
import scipy.ndimage as ndimage

if __name__ == "__main__":
    
    filepath = r'data/test_220621'
    filename = r'benchmarkHR'
    
    input_filepath = f'{filepath}/{filename}.h5'

    # colname = 'mag_v'
    # colname = 'u'
    colname = 'mask'

    idx = 0
    with h5py.File(input_filepath, 'r') as hf:
        img = np.asarray(hf.get(colname)[idx])

    # img = ndimage.zoom(img, 2, order=3)

    msv.multi_slice_viewer(img, slice_axis=1)