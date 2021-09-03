import numpy as np
import h5py
import matplotlib.pyplot as plt
from icecream import ic

def regression(hr_filepath, pred_filepath, ts=26, all=True):
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
    
    peakvx = max(pvx)
    peakvy = max(pvy)
    peakvz = max(pvz)
    peakvm = max(pvm)

    ic(peakvx, peakvy, peakvz, peakvm)
    
    errorx = vxs - pvx
    errory = vys - pvy
    errorz = vzs - pvz
    errorm = vms - pvm

    sepvx = np.std(errorx)
    sepvy = np.std(errory)
    sepvz = np.std(errorz)
    sepvm = np.std(errorm)

    ic(sepvx, sepvy, sepvz, sepvm)

    i = 0
    subplot_num = 421
    titles = ["vx", "vy", "vz", "vm"]
    length = len(vxs)
    if all:
        for vi, pvi, sepvi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm], [sepvx, sepvy, sepvz, sepvm]):
            xhrsr = (vi+pvi)/2
            m, b = np.polyfit(xhrsr, pvi, 1)
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
            
    return np.sqrt(np.mean((errorx)**2)), np.sqrt(np.mean((errory)**2)), np.sqrt(np.mean((errorz)**2))

if __name__ == '__main__':

    hr_filepath = "data/Unused Geometries/trainG4HR.h5"
    pred_filepath = "result/base4x_result.h5"

    rmsex, rmsey, rmsez = regression(hr_filepath, pred_filepath)
    ic(rmsex, rmsey, rmsez)