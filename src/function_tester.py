from operator import sub
import numpy as np
import h5py
import matplotlib.pyplot as plt

def rotate(u, v, w, plane, theta):
    x = [u, v, w]

    if plane == 1:
        ax = (0, 1)
        rmat = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]

    elif plane == 2:
        ax = (0, 2)
        rmat = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]

    elif plane == 3:
        ax = (1, 2)
        rmat = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]

    else:
        return u, v, w
        
    x = np.matmul(rmat, x)

    return x[0], x[1], x[2]

def regression(hr_filepath, pred_filepath, ts=30, all=True):
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
    
    sepvx = np.std(pvx)
    sepvy = np.std(pvy)
    sepvz = np.std(pvz)
    sepvm = np.std(pvm)

    i = 0
    subplot_num = 421
    titles = ["vx", "vy", "vz", "vm"]
    length = len(vxs)
    if all:
        for vi, pvi, sepvi in zip([vxs, vys, vzs, vms], [pvx, pvy, pvz, pvm], [sepvx, sepvy, sepvz, sepvm]):
            xhrsr = (vi+pvi)/2
            m, b = np.polyfit(xhrsr, pvi, 1)
            plt.subplot(subplot_num+i), plt.scatter(xhrsr, pvi, s=0.2, alpha=0.01, c='black'), plt.title(titles[i//2])
            plt.subplot(subplot_num+i), plt.plot(xhrsr, m*xhrsr + b, alpha=0.5), plt.title(titles[i//2])
            i += 1
            plt.subplot(subplot_num+i), plt.scatter(xhrsr, pvi-vi, s=0.2, alpha=0.01, c='black'), plt.title("sfwe")
            plt.subplot(subplot_num+i), plt.plot(xhrsr, np.repeat(sepvi, length), '--', alpha=0.5), plt.title(titles[i//2])
            plt.subplot(subplot_num+i), plt.plot(xhrsr, np.repeat(-sepvi, length), '--', alpha=0.5), plt.title(titles[i//2])
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
            
    return np.mean((vxs-pvx)**2) + np.mean((vys-pvy)**2) + np.mean((vzs-pvz)**2)

if __name__ == '__main__':
    u = np.array([2, 2.1, 1.9, 2.2, 2.])
    v = np.array([1.5, 1.6, 1.4, 1.7, 1.2])
    w = np.array([0., 0.1, -0.1, -0.2, 0.])

    # u, v, w = rotate(u, v, w, 1, np.pi*1.5)
    # print(u, v, w)

    hr_filepath = "data/trainG4HR.h5"
    pred_filepath = "result/cspnet_noAliasing_G5_G4.h5"

    total_mspe = regression(hr_filepath, pred_filepath)
    print(total_mspe)