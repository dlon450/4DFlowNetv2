import numpy as np
import h5py
from h5functions import save_to_h5
import os 

if __name__ == "__main__":
    output_name = "data/trainG6HR.h5"
    u = np.zeros((72, 392, 52, 52))
    v = np.zeros((72, 392, 52, 52))
    w = np.zeros((72, 392, 52, 52))
    dx_o = np.zeros((72, 3))
    origin_o = np.zeros((72, 3))
    mask = np.zeros((1, 392, 52, 52))

    data_dir = 'data/test_020721'
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith('.csv'), all_files))
    data_files = [data_dir + '/' + d for d in data_files]
    files = data_files

    i = 0
    first = True
    for fn in files:
        if first:
            first = False
            continue
        with h5py.File(fn, mode='r') as hf:
            dx = np.asarray(hf['dx'])
            origin = np.asarray(hf['origin'])
            vx1 = np.asarray(hf['u'])
            vy1 = np.asarray(hf['v'])
            vz1 = np.asarray(hf['w'])
            if first:
                mask = np.asarray(hf['mask'])
                first = False
        n = vx1.shape[0]
        u[i:i+n] = vx1
        v[i:i+n] = vy1
        w[i:i+n] = vz1
        dx_o[i:i+n] = dx
        origin_o[i:i+n] = origin

        i += n
    
    save_to_h5(output_name, f"dx", dx_o, expand=False)
    save_to_h5(output_name, f"origin", origin_o, expand=False)
    save_to_h5(output_name, f"u", u, expand=False)
    save_to_h5(output_name, f"v", v, expand=False)
    save_to_h5(output_name, f"w", w, expand=False)
    # save_to_h5(output_name, f"mask", mask, expand=False)