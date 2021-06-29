from threading import active_count
import tensorflow as tf
import numpy as np
import h5py
from scipy import ndimage
import random
from icecream import ic

class PatchHandler3D():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = {True: res_increase*patch_size, False: patch_size}
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        # for 45' angles
        actual_size = int(patch_size * 1.5)
        self.actual_size = {True: self.res_increase*actual_size, False: actual_size}
        offset = (actual_size - patch_size) // 2
        self.offset = {True: self.res_increase*offset, False: offset}

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds

    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def check_edge(self, start, hr):
        '''
        Check if the start is outside of the image
        '''
        if start < 0:
            pad = np.abs(start)
            actual_size = self.actual_size[hr] - pad
            start = 0
        else:
            pad = 0
            actual_size = self.actual_size[hr]

        return start, pad, actual_size

    def get_extended_patch_roi(self, idx, x_start, y_start, z_start, hr=True):
        '''
        Return the larger patch and padding
        '''
        x_start, pad_x, actual_size_x = self.check_edge(x_start - self.offset[hr], hr)
        y_start, pad_y, actual_size_y = self.check_edge(y_start - self.offset[hr], hr)
        z_start, pad_z, actual_size_z = self.check_edge(z_start - self.offset[hr], hr)

        roi = np.index_exp[idx, x_start:x_start + actual_size_x, y_start:y_start+actual_size_y, z_start:z_start+actual_size_z]

        if hr:
            mask_roi = np.index_exp[0, x_start:x_start+actual_size_x, y_start:y_start+actual_size_y, z_start:z_start+actual_size_z]
            return roi, mask_roi, (pad_x, pad_y, pad_z)
        
        return roi, (pad_x, pad_y, pad_z)

    def ensure_patch_size(self, u, pad_x, pad_y, pad_z, hr=True):
        # left padding
        u = np.pad(u, ((pad_x, 0), (pad_y, 0), (pad_z, 0)))
        # print('after', u.shape)

        # calculate if the image is smaller than the size
        rpad_x = self.actual_size[hr] - u.shape[0]
        rpad_y = self.actual_size[hr] - u.shape[1]
        rpad_z = self.actual_size[hr] - u.shape[2]

        # pad it to the right or bottom
        u = np.pad(u, ((0, rpad_x), (0, rpad_y), (0, rpad_z)))
        
        return u

    def crop_to_patch_size(self, u_new, hr=True):
        end = self.offset[hr] + self.patch_size[hr]
        return u_new[self.offset[hr]: end, self.offset[hr]: end, self.offset[hr]: end]

    def load_patches_from_index_file(self, indexes):
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])

        # patch_size = self.patch_size
        # hr_patch_size = self.patch_size * self.res_increase
        
        # ============ get the patch =============
        # patch_index  = np.index_exp[idx, x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
        # hr_patch_index = np.index_exp[idx, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]
        # mask_index = np.index_exp[0, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]

        patch_index, (pad_x, pad_y, pad_z) = self.get_extended_patch_roi(idx, x_start, y_start, z_start, hr=False)
        hr_patch_index, mask_index, (hrpad_x, hrpad_y, hrpad_z) = self.get_extended_patch_roi(idx, x_start*self.res_increase, y_start*self.res_increase, z_start*self.res_increase)
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index)

        # make sure the patch size match the extended size (handles edge cases)
        # print('before ensure',u_patch.shape)
        # LOWRES
        u_patch = self.ensure_patch_size(u_patch, pad_x, pad_y, pad_z, hr=False)
        v_patch = self.ensure_patch_size(v_patch, pad_x, pad_y, pad_z, hr=False)
        w_patch = self.ensure_patch_size(w_patch, pad_x, pad_y, pad_z, hr=False)
        
        mag_u_patch = self.ensure_patch_size(mag_u_patch, pad_x, pad_y, pad_z, hr=False)
        mag_v_patch = self.ensure_patch_size(mag_v_patch, pad_x, pad_y, pad_z, hr=False)
        mag_w_patch = self.ensure_patch_size(mag_w_patch, pad_x, pad_y, pad_z, hr=False)

        # HIRES
        u_hr_patch = self.ensure_patch_size(u_hr_patch, hrpad_x, hrpad_y, hrpad_z)
        v_hr_patch = self.ensure_patch_size(v_hr_patch, hrpad_x, hrpad_y, hrpad_z)
        w_hr_patch = self.ensure_patch_size(w_hr_patch, hrpad_x, hrpad_y, hrpad_z)

        mask_patch = self.ensure_patch_size(mask_patch, hrpad_x, hrpad_y, hrpad_z)
        # print('after ensure', u_patch.shape)

        # ============ apply rotation ============ 
        if is_rotate > 0:
            additional = 0
            if random.random() < 0.2:
                additional += [-1, 1][random.randint(0, 1)] * np.pi/4

            u_patch, v_patch, w_patch = self.apply_rotation(u_patch, v_patch, w_patch, rotation_degree_idx, rotation_plane, True, additional, 1)
            u_hr_patch, v_hr_patch, w_hr_patch = self.apply_rotation(u_hr_patch, v_hr_patch, w_hr_patch, rotation_degree_idx, rotation_plane, True, additional, 1)
            mag_u_patch, mag_v_patch, mag_w_patch = self.apply_rotation(mag_u_patch, mag_v_patch, mag_w_patch, rotation_degree_idx, rotation_plane, False, additional, 1)
            mask_patch = self.rotate_object(mask_patch, rotation_degree_idx, rotation_plane, additional, 0)
            
        u_patch = self.crop_to_patch_size(u_patch, hr=False)
        v_patch = self.crop_to_patch_size(v_patch, hr=False)
        w_patch = self.crop_to_patch_size(w_patch, hr=False)

        mag_u_patch = self.crop_to_patch_size(mag_u_patch, hr=False)
        mag_v_patch = self.crop_to_patch_size(mag_v_patch, hr=False)
        mag_w_patch = self.crop_to_patch_size(mag_w_patch, hr=False)

        u_hr_patch = self.crop_to_patch_size(u_hr_patch)
        v_hr_patch = self.crop_to_patch_size(v_hr_patch)
        w_hr_patch = self.crop_to_patch_size(w_hr_patch)

        mask_patch = self.crop_to_patch_size(mask_patch)

        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    
    def rotate_object(self, img, rotation_idx, plane_nr, additional, interp_order):
        if plane_nr==1:
            ax = (0,1)
        elif plane_nr==2:
            ax = (0,2)
        elif plane_nr==3:
            ax = (1,2)
        else:
            # Unspecified rotation plane, return original
            return img

        img = ndimage.rotate(img, 90*rotation_idx + additional, axes=ax, order=interp_order)
        return img

    def apply_rotation(self, u, v, w, rotation_idx, plane_nr, is_phase_image, additional, interp_order):

        rotation_angle = np.pi*rotation_idx/2 + additional
        u, v, w = rotate(u, v, w, plane_nr, rotation_angle, is_phase_image, interp_order)

        return u, v, w         

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            mask = (mask >= self.mask_threshold) * 1.
            
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = hl.get(self.venc_colnames[i])[idx]

                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)

        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc

# ============== Rotation and flip ==============
def rotate180_3d(u, v, w, plane=1, is_phase_img=True):
    """
        Rotate 180 degrees to introduce negative values
        xyz Axis stays the same
    """
    if plane==1:
        # Rotate on XY, y*-1, z*-1
        ax = (0,1)
        if is_phase_img:
            v *= -1
            w *= -1
    elif plane==2:
        # Rotate on XZ, x*-1, z*-1
        ax = (0,2)
        if is_phase_img:
            u *= -1
            w *= -1
    elif plane==3:
        # Rotate on YZ, x*-1, y*-1
        ax = (1,2)
        if is_phase_img:
            u *= -1
            v *= -1
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 180 deg rotation
    u = np.rot90(u, k=2, axes=ax)
    v = np.rot90(v, k=2, axes=ax)
    w = np.rot90(w, k=2, axes=ax)    

    return u,v,w

def rotate90(u, v, w, plane, k, is_phase_img=True):
    """
        Rotate 90 (k=1) or 270 degrees (k=3)
        Introduce axes swapping and negative values
    """
    if plane==1:
        
        ax = (0,1)
        
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XY, swap Z to Y +, Y to Z -
            temp = v
            v = w
            w = temp 
            if is_phase_img:
                w *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XY, swap Z to Y -, Y to Z +
            temp = v
            v = w
            if is_phase_img:
                w *= -1
            w = temp

            

    elif plane==2:
        ax = (0,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XZ, swap X to Z +, Z to X -
            temp = w
            w = u
            u = temp 
            if is_phase_img:
                u *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XZ, swap X to Z -, Z to X +
            temp = w
            w = u
            if is_phase_img:
                w *= -1
            u = temp
        
    elif plane==3:
        ax = (1,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on YZ, swap X to Y +, Y to X -
            temp = v
            v = u
            u = temp
            if is_phase_img:
                u *= -1
        elif k ==3:
            # =================== ROTATION 270 =================== 
            # Rotate on YZ, swap X to Y -, Y to X +
            temp = v
            v = u
            if is_phase_img:
                v *= -1
            u = temp
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 90 or 270 deg rotation
    u = np.rot90(u, k=k, axes=ax)
    v = np.rot90(v, k=k, axes=ax)
    w = np.rot90(w, k=k, axes=ax)    

    return u,v,w

def rotate(u, v, w, plane, theta, is_phase_img=True, interp_order=1):
    '''
    Rotate by theta (radians) counterclockwise
    '''

    if plane == 1:
        ax = (0, 1)
        rmat = np.array([
            [1,             0,              0], 
            [0, np.cos(theta), -np.sin(theta)], 
            [0, np.sin(theta), np.cos(theta)]])
    
    elif plane == 2:
        ax = (0, 2)
        rmat = np.array([
            [ np.cos(theta), 0, np.sin(theta)], 
            [             0, 1,             0], 
            [-np.sin(theta), 0, np.cos(theta)]])
    
    elif plane == 3:
        ax = (1, 2)
        rmat = np.array([
            [np.cos(theta), -np.sin(theta), 0], 
            [np.sin(theta),  np.cos(theta), 0], 
            [            0,              0, 1]])
    
    else:
        return u, v, w
    
    degrees = theta*180/np.pi 

    if is_phase_img:
        original_shape = u.shape
        vmap = np.stack((u, v, w))
        vmap = np.reshape(vmap, (3, -1)) # reshape to 3 vectors
        rotated = rmat.dot(vmap)
        u_r, v_r, w_r = rotated[0], rotated[1], rotated[2]

        u_r = np.reshape(u_r, original_shape)
        v_r = np.reshape(v_r, original_shape)
        w_r = np.reshape(w_r, original_shape)

        u_r = ndimage.rotate(u_r, angle=degrees, axes=ax, reshape=False, order=interp_order)
        w_r = ndimage.rotate(w_r, angle=degrees, axes=ax, reshape=False, order=interp_order)
        v_r = ndimage.rotate(v_r, angle=degrees, axes=ax, reshape=False, order=interp_order)
    else:
        u_r = ndimage.rotate(u, angle=degrees, axes=ax, reshape=False, order=interp_order)
        w_r = ndimage.rotate(w, angle=degrees, axes=ax, reshape=False, order=interp_order)
        v_r = ndimage.rotate(v, angle=degrees, axes=ax, reshape=False, order=interp_order)

    return u_r, w_r, v_r

def flip(u, v, w, plane):
    '''
    Flip horizontally (plane=1) or vertically (plane=2)
    '''
    idx = plane - 1
    flip_mat = np.identity(3)
    flip_mat[idx][idx] = -1

    u_f = np.matmul(flip_mat, u)
    v_f = np.matmul(flip_mat, v)
    w_f = np.matmul(flip_mat, w)

    return u_f, v_f, w_f

def scale(u, v, w):
    '''
    Scale velocities
    '''
    sx = random.uniform(0.5, 1.5)
    sy = random.uniform(0.5, 1.5)
    sz = random.uniform(0.5, 1.5)

    s = np.zeros((3,3))
    s[0][0] = sx
    s[1][1] = sy
    s[2][2] = sz

    u_s = np.matmul(s, u)
    v_s = np.matmul(s, v)
    w_s = np.matmul(s, w)

    return u_s, v_s, w_s