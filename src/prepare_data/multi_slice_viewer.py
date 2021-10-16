"""
    Original source: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
"""

import matplotlib.pyplot as plt
import numpy as np

def update_title(title, ax, total_slice):
    title_str = f' << Prev (J) - {viewer_title} {ax.index}/{total_slice-1} - Next (K) >>'
    ax.set_title(title_str)

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, slice_axis=0, color='jet', show_colorbar=True, clim=None, show_grid=False, title='slice', save=False, savedir = '.' ):
    global viewer_title
    global save_mode, save_dir
    save_dir = savedir
    save_mode = save
    viewer_title = title

    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()

    assert slice_axis < 3, "Axis not supported, slice_axis must be between 0 and 2"

    if slice_axis == 1:
        volume = np.transpose(volume, (1, 0, 2))
    elif slice_axis == 2:
        volume = np.transpose(volume, (2, 0, 1))
 
    ax.volume = volume
    ax.index = volume.shape[0] // 2

    # if clim is None:
    im1 = ax.imshow(volume[ax.index], cmap=color, clim=clim)

    if show_colorbar:
        fig.colorbar(im1)
    
    update_title(title, ax, volume.shape[0])

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.grid(show_grid)
    plt.axis('off')
    if save_mode:
        plt.savefig('{}/{}{}.png'.format(save_dir, 'img', ax.index), dpi=150)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
    
    if save_mode:
        plt.savefig('{}/{}{}.png'.format(save_dir, 'img', ax.index), dpi=150)

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    
    update_title(viewer_title, ax, volume.shape[0])
    

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

    update_title(viewer_title, ax, volume.shape[0])
    