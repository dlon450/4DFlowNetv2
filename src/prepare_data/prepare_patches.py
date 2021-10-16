import numpy as np
import h5py
import PatchData as pd

def load_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = np.min([len(hdf5.get(i)) for i in ["u", "v", "w"]])
        print(hdf5['w'])
    indexes = np.arange(data_nr)
    print("Dataset: {} rows".format(len(indexes)))
    return indexes

def generate_patches_csv(lr_file, hr_file, output_filename, patch_size=16, n_patch=23, n_empty_patch_allowed=0, all_rotation=False, mask_threshold=0.4, minimum_coverage = 0.2):
    # Load the data
    input_filepath = f'{base_path}/{lr_file}'
    file_indexes = load_data(input_filepath)

    # Prepare the CSV output
    pd.write_header(output_filename)

    # because the data is homogenous in 1 table, we only need the first data
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        mask = np.asarray(hdf5['mask'][0])
    # We basically need the mask on the lowres data, the patches index are retrieved based on the LR data.
    print("Overall shape", mask.shape)

    # Do the thresholding
    binary_mask = (mask >= mask_threshold) * 1
    # print(binary_mask)
    print(file_indexes)
    # Generate random patches for all time frames
    for index in file_indexes:
        print('Generating patches for row', index)
        pd.generate_random_patches(lr_file, hr_file, output_filename, index, n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)
    print(f'Done. File saved in {output_filename}')

if __name__ == "__main__": 
    patch_size = 12 # Patch size, this will be checked to make sure the generated patches do not go out of bounds
    n_patch = 10 # number of patch per time frame
    n_empty_patch_allowed = 0 # max number of empty patch per frame
    all_rotation = False # When true, include 90,180, and 270 rotation for each patch. When False, only include 1 random rotation.
    mask_threshold = 0.4 # Threshold for non-binary mask 
    minimum_coverage = 0.2 # Minimum fluid region within a patch. Any patch with less than this coverage will not be taken. Range 0-1
    args = [patch_size, n_patch, n_empty_patch_allowed, all_rotation, mask_threshold, minimum_coverage]

    base_path = 'data'
    # lr_files = ['trainLR.h5', 'validationLR.h5', 'benchmarkLR.h5']
    # hr_files = ['trainHR.h5', 'validationHR.h5', 'benchmarkHR.h5']
    # output_filenames = [f'{base_path}/train.csv', f'{base_path}/validate.csv', f'{base_path}/benchmark.csv']

    # for lr_file, hr_file, output_filename in zip(lr_files, hr_files, output_filenames):
    #     generate_patches_csv(lr_file, hr_file, output_filename, *args)

    files_dict = {
        'train.csv': [['trainG1LR.h5','trainG2LR.h5', 'trainG3LR.h5', 'trainG5LR.h5'], ['trainG1HR.h5', 'trainG2HR.h5', 'trainG3HR.h5', 'trainG5HR.h5']],
        'validation.csv': [['trainG4LR.h5'], ['trainG4HR.h5']]
    }

    i = 20
    files_dict = {
        'train{}.csv'.format(i): [['trainG{}LR.h5'.format(i)], ['trainG{}HR.h5'.format(i)]],
    }

    for output_filename, files in files_dict.items():
        for lr_file, hr_file in zip(files[0], files[1]):
            generate_patches_csv(lr_file, hr_file, f'{base_path}/{output_filename}', *args)