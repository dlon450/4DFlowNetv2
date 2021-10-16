import numpy as np
import h5py
import matplotlib.pyplot as plt
from time import perf_counter as pc
from skimage.metrics import structural_similarity as ssim
from os import listdir
from icecream import ic

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-5): 
    '''
    Calculate the mean absolute percentage error between true and predicted values
    '''
    y_true, y_pred = np.array(y_true*100), np.array(y_pred*100)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def maape(actual: np.ndarray, predicted: np.ndarray, eps=1e-10):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + eps))))

def regression(hr_filepath, pred_filepath, ts=26, all_plots=True):
    '''
    Plot regression plots for each velocity and return the RMSE
    '''
    with h5py.File(hr_filepath, 'r') as hf:
        vx = np.asarray(hf['u'])[ts]
        vy = np.asarray(hf['v'])[ts]
        vz = np.asarray(hf['w'])[ts]
        vm = np.sqrt(vx**2+vy**2+vz**2)
        mask = np.asarray(hf['mask'])[0]
    
    with h5py.File(pred_filepath, 'r') as hf:
        pvx = np.asarray(hf['u'])[ts]
        pvy = np.asarray(hf['v'])[ts]
        pvz = np.asarray(hf['w'])[ts]
        pvm = np.sqrt(pvx**2+pvy**2+pvz**2)
    
    mask = mask[:,:48,:48]
    subset = mask.ravel() == 1
    vxs = vx[:,:48,:48].ravel()[subset]
    vys = vy[:,:48,:48].ravel()[subset]
    vzs = vz[:,:48,:48].ravel()[subset]
    vms = vm[:,:48,:48].ravel()[subset]

    pvx = pvx.ravel()[subset]
    pvy = pvy.ravel()[subset]
    pvz = pvz.ravel()[subset]
    pvm = np.sqrt(pvx**2+pvy**2+pvz**2) 
    
    # mape = [mean_absolute_percentage_error(vi, pi) for vi, pi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm])]
    maape_speed = maape(vxs, pvx)
    ic(maape_speed)

    peakvx = max(vxs)
    peakvy = max(vys)
    peakvz = max(vzs)
    peakvm = max(vms)

    ic(peakvx, peakvy, peakvz, peakvm)
    
    errorx = vxs - pvx
    errory = vys - pvy
    errorz = vzs - pvz
    errorm = vms - pvm

    sepvx = np.std(errorx)
    sepvy = np.std(errory)
    sepvz = np.std(errorz)
    sepvm = np.std(errorm)

    i = 0
    subplot_num = 421
    titles = ["vx", "vy", "vz", "vm"]
    length = len(vxs)
    if all_plots:
        for vi, pvi, sepvi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm], [sepvx, sepvy, sepvz, sepvm]):
            xhrsr = (vi+pvi)/2
            m, b = np.polyfit(xhrsr, pvi, 1)
            ic(np.round(m, 4), np.round(b, 4))
            plt.subplot(subplot_num+i), plt.scatter(xhrsr, pvi, s=0.2, alpha=0.1, c='black')#, plt.title(titles[i//2])
            plt.subplot(subplot_num+i), plt.plot(xhrsr, m*xhrsr + b, alpha=0.5)#, plt.title(titles[i//2])
            i += 1
            plt.subplot(subplot_num+i), plt.scatter(xhrsr, pvi-vi, s=0.2, alpha=0.1, c='black'), plt.title("")
            plt.subplot(subplot_num+i), plt.plot(xhrsr, np.repeat(sepvi, length), '--', alpha=0.5)
            plt.subplot(subplot_num+i), plt.plot(xhrsr, np.repeat(-sepvi, length), '--', alpha=0.5)
            i += 1

        plt.tight_layout()
        plt.show()
    
    else:
        for vi, pvi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm]):
            xhrsr = (vi+pvi)/2
            m, b = np.polyfit(xhrsr, pvi, 1)
            plt.scatter(xhrsr, pvi, s=0.2, alpha=0.01, c='black'), plt.title(titles[i//2])
            plt.plot(xhrsr, m*xhrsr + b, alpha=0.5), plt.title(titles[i//2])
            plt.show()  

        for vi, pvi, sepvi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm], [sepvx, sepvy, sepvz, sepvm]):
            xhrsr = (vi+pvi)/2
            plt.scatter(xhrsr, pvi-vi, s=0.2, alpha=0.01, c='black'), plt.title("sfwe")
            plt.plot(xhrsr, np.repeat(sepvi, length), '--', alpha=0.5), plt.title(titles[i//2])
            plt.plot(xhrsr, np.repeat(-sepvi, length), '--', alpha=0.5), plt.title(titles[i//2])
            plt.show()
            
    return [np.round(e, 5) for e in [np.sqrt(np.mean((errorx)**2)), np.sqrt(np.mean((errory)**2)), np.sqrt(np.mean((errorz)**2))]], [np.round(se, 5) for se in [sepvx, sepvy, sepvz, sepvm]]

def mean_structural_similarity(hr_filepath, pred_filepath):
    with h5py.File(hr_filepath, 'r') as hf:
        vx = np.asarray(hf['u'])[:,:,:48,:48]
        vy = np.asarray(hf['v'])[:,:,:48,:48]
        vz = np.asarray(hf['w'])[:,:,:48,:48]
    
    with h5py.File(pred_filepath, 'r') as hf:
        pvx = np.asarray(hf['u'])
        pvy = np.asarray(hf['v'])
        pvz = np.asarray(hf['w'])
    
    ic(vx.shape, pvx.shape)

    tic = pc()
    ssimx = ssim(vx, pvx, data_range=pvx.max()-pvx.min())
    toc = pc()
    elapsed = toc - tic
    ssimy = ssim(vy, pvy, data_range=pvy.max()-pvy.min())
    ssimz = ssim(vz, pvz, data_range=pvz.max()-pvz.min())

    
    return [ssimx, ssimy, ssimz], elapsed

if __name__ == '__main__':

    hr_filepath = "data/trainG5HR.h5"
    pred_folder = "result/five_geom/no_aliasing"
    ic(hr_filepath, pred_folder)
    files = listdir(pred_folder)
    files = ['base4x_5geom.h5','resnet.h5', 'csp.h5', 'dense_sse.h5']

    with plt.style.context('bmh'):
        for file in files:
            pred_filepath = '{}/{}'.format(pred_folder, file)
            ic(pred_filepath)
            rmse, se = regression(hr_filepath, pred_filepath)
            ic(rmse, se)

    # ssim_all, elapsed_x = mean_structural_similarity(hr_filepath, pred_filepath)
    # ic(ssim_all, elapsed_x)