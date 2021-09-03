import numpy as np
import scipy.interpolate as sc
from icecream import ic
import time
import os
from random import shuffle
import h5py
import multi_slice_viewer as msv
from scipy.spatial import Delaunay

from scipy.spatial import KDTree
from h5functions import save_to_h5

class CFDResult():
    def __init__(self, filepath, rounding):
        self.read_csv(filepath, rounding)

    def read_csv(self, filepath, rounding):
        """
            Read xyz coordinates and pressure from CFD results (.csv)
            XYZ is converted to mm
        """
        names = ['xcoordinate','ycoordinate','zcoordinate','pressure','xvelocity','yvelocity','zvelocity']
        arr = np.genfromtxt(filepath, delimiter=',', names=names, skip_header=6)
        
        x, y, z = arr['xcoordinate'], arr['ycoordinate'], arr['zcoordinate']
        
        # print(wss.shape)
        self.pressure = arr['pressure']
        # convert this to mm
        self.x = np.round(x * 1000, rounding)
        self.y = np.round(y * 1000, rounding)
        self.z = np.round(z * 1000, rounding)

        self.vx = arr['xvelocity']
        self.vy = arr['yvelocity']
        self.vz = arr['zvelocity']
        # self.speed = arr['velocitymagnitude']
        self.speed = np.sqrt(self.vx**2+self.vy**2+self.vz**2)
        
def get_minmax_arr(arr, skip_x):
    min_x, max_x = np.min(arr), np.max(arr)
    min_x, max_x = np.floor(min_x), np.ceil(max_x)
    # ic(min_x, max_x)

    min_x = min_x - skip_x
    max_x = max_x + skip_x
    x_arr = np.arange(min_x, max_x , skip_x)
    # ic(x_arr, x_arr.shape)
    return x_arr, min_x, max_x 

def split_train_test_val(data_dir, split=0.7):
    all_files = os.listdir(os.path.abspath(data_dir))
    data_files = list(filter(lambda file: file.endswith('.csv'), all_files))
    data_files = [data_dir + '/' + d for d in data_files]
    # shuffle(data_files)

    split_index_val = int(np.floor(len(data_files) * split))
    split_index_bench = int(np.floor(len(data_files) * (split + 1) / 2))
    training = data_files[:split_index_val]
    validation = data_files[split_index_val:split_index_bench]
    benchmark = data_files[split_index_bench:]
    return training, validation, benchmark

def convert_to_h5(file_list, output_name, dx):
    first = False
    current_index = 1420
    for velocity_file in file_list:
        if current_index >= 2080:
            print(velocity_file)
            # Load the velocity data
            vel_cfd_res = CFDResult(velocity_file, 4)

            # Reshape xyz coordinates
            v_coords = np.stack((vel_cfd_res.x, vel_cfd_res.y, vel_cfd_res.z), axis=-1)
        
            # get min max coordinates
            x_arr, min_x, max_x = get_minmax_arr(v_coords[:,0], dx)
            y_arr, min_y, max_y = get_minmax_arr(v_coords[:,1], dx)
            z_arr, min_z, max_z = get_minmax_arr(v_coords[:,2], dx)

            xx, yy, zz = np.mgrid[min_x:max_x: dx, min_y:max_y:dx, min_z:max_z:dx]
            ic(len(yy), len(zz))
            yy = np.asarray(yy)
            zz = np.asarray(zz)
            ic(xx.shape, yy.shape, zz.shape)

            # msv.multi_slice_viewer(interpolate_mask(v_coords, xx, yy, zz), slice_axis=1)
            # break
            tri = Delaunay(v_coords)

            # Prepare interpolator
            print("Preparing velocity interpolation")
            interpolator_vx = sc.LinearNDInterpolator(tri, vel_cfd_res.vx)
            interpolator_vy = sc.LinearNDInterpolator(tri, vel_cfd_res.vy)
            interpolator_vz = sc.LinearNDInterpolator(tri, vel_cfd_res.vz)

            print("Interpolating...")
            vx1 = interpolator_vx(xx,yy,zz)
            vy1 = interpolator_vy(xx,yy,zz)
            vz1 = interpolator_vz(xx,yy,zz)

            # ic(vx1.shape)
            vx1 = np.nan_to_num(vx1)
            vy1 = np.nan_to_num(vy1)
            vz1 = np.nan_to_num(vz1)

            outname = output_name + str(current_index)
            save_to_h5(outname, f"dx", (dx,dx,dx))
            save_to_h5(outname, f"origin", (min_x,min_y,min_z))
            save_to_h5(outname, f"u", vx1)
            save_to_h5(outname, f"v", vy1)
            save_to_h5(outname, f"w", vz1)

        current_index += 10

        if first:
            print("Interpolating mask...")
            save_to_h5(output_name, f"maski", interpolate_mask(v_coords, xx, yy, zz))
            first = False
        
def interpolate_mask(v_coords, xx, yy, zz):
    # --- get mask ---
    tree = KDTree(v_coords, leafsize=10)
    probe1 = np.stack((xx, yy, zz), axis=-1)
    distances, ndx = tree.query(probe1, k=1, distance_upper_bound=0.5)
    # print(distances.shape)

    # consider point as a mask point when there is a point closer than half of dx
    mask = distances <= (100)
    return mask

def create_mask(filename, threshold=0.0005, interval=None):
    with h5py.File(filename, mode = 'r' ) as hf:
        vx1 = np.asarray(hf['u'][35])
        vy1 = np.asarray(hf['v'][35])
        vz1 = np.asarray(hf['w'][35])

        maski = np.asarray(hf['maski'][0])

    skip = 20
    mid = skip + np.argmin(maski[skip:maski.shape[0]-skip].sum(axis=1).sum(axis=1))

    mask = np.zeros(vx1.shape, dtype='int')
    a = np.array([1., 50., 10.])

    if interval is None:
        # thresholding using the average velocity
        for i in range(mask.shape[0]):
            matrix = a[0]*vx1[i]**2 + a[1]*vy1[i]**2 + a[2]*vz1[i]**2
            mask[i] = np.where(matrix > np.true_divide(matrix.sum(), 1+5*(matrix >= 1e-06).sum()), 1, 0)
        
        interval = [int(mid - 0.05*mask.shape[0]), int(mid + 0.05*mask.shape[0])]
        interval = [110, 143]

        # thresholding outer section with constant
        matrix = vx1**2 + vy1**2 + vz1**2
        mask_outer = np.where(matrix > threshold, 1, 0)
        mask = np.concatenate((mask_outer[:interval[0]], mask[interval[0]:interval[1]] + 1, mask_outer[interval[1]:]))
        msv.multi_slice_viewer(mask, slice_axis=1)
        mask = mask*maski
        msv.multi_slice_viewer(mask, slice_axis=1)

    else:
        # thresholding with constant
        matrix = vx1**2 + vy1**2 + vz1**2
        mask = np.where(matrix > threshold, 1, 0)

        # thresholding using the average velocity
        for i in range(interval[0], interval[1]):
            matrix = a[0]*vx1[i]**2 + a[1]*vy1[i]**2 + a[2]*vz1[i]**2
            mask[i] = np.where(matrix > np.true_divide(matrix.sum(), 1+5*(matrix >= 1e-06).sum()), 1, 0)

        # find the thinnest part of geometry
        skip = 10
        mid = skip + np.argmin(mask[skip:].sum(axis=1).sum(axis=1))

    # ensuring consistency in structure/geometry
    updated_mask = np.concatenate((np.cumsum(mask[:mid][::-1], axis=0)[::-1], np.cumsum(mask[mid:], axis=0)))
    updated_mask = np.where(updated_mask != 0, 1, 0)
    msv.multi_slice_viewer(updated_mask, slice_axis=0)
    save_to_h5(filename, f"mask", updated_mask)

if __name__ == "__main__":

    data_dir = fr'CFD Output/Model8'
    dx = 0.2 # grid spacing
    output_dir = fr'data'

    np.random.seed(346511053)
    training, validation, benchmark = split_train_test_val(data_dir, split=1)

    # convert_to_h5(benchmark, os.path.join(output_dir,'benchmarkHR.h5'), dx)
    # convert_to_h5(training, os.path.join(output_dir,'trainG8HR.h5'), dx)
    # convert_to_h5(validation, os.path.join(output_dir,'validationHR.h5'), dx)
    
    create_mask(os.path.join(output_dir, 'trainG3HR.h5'))
    # create_mask(os.path.join(output_dir, 'validationHR.h5'))
    # create_mask(os.path.join(output_dir, 'benchmarkHR.h5')) 