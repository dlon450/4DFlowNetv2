import numpy as np
import h5py
import matplotlib.pyplot as plt
from icecream import ic
import warnings
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def read_h5(filepath, i, o, b=48, has_mask=True):
    
    mask = None
    with h5py.File(filepath, 'r') as hf:
        vx = np.asarray(hf['u'])[:,i:o,:b,:b]
        vy = np.asarray(hf['v'])[:,i:o,:b,:b]
        vz = np.asarray(hf['w'])[:,i:o,:b,:b]
        if has_mask:
            mask = np.asarray(hf['mask'])[0,i:o,:b,:b]
        vm = np.sqrt(vx**2+vy**2+vz**2)
    
    return vm, mask


def plot_pressure(hr_filepath, pred_filepath, lr_filepath, downsample=4, inlet=108, outlet=128, bounds=48, record=None):
    
    vm, hmask = read_h5("data/trainG4HR.h5", inlet, outlet)
    _, mask = read_h5(hr_filepath, inlet*downsample//4, outlet*downsample//4, bounds*downsample//4)
    pvm, _ = read_h5(pred_filepath, inlet*downsample//4, outlet*downsample//4, bounds*downsample//4, has_mask=False)
    lvm, lmask = read_h5(lr_filepath, inlet//4, outlet//4, bounds//4)
    
    # vm_pressure = np.array([4*np.max(v*hmask)**2 for v in vm])
    # pvm_pressure = np.array([4*np.max(v*mask)**2 for v in pvm])
    # lvm_pressure = np.array([4*np.max(v*lmask)**2 for v in lvm])
    
    vm_maxt = np.array([np.max(v*hmask) for v in vm])
    pvm_maxt = np.array([np.max(v*mask) for v in pvm])
    lvm_maxt = np.array([np.max(v*lmask) for v in lvm])

    hsubset = hmask.ravel() == 1
    psubset = mask.ravel() == 1
    lsubset = lmask.ravel() == 1    
    #boxplot(vm, lvm, hsubset, lsubset, vm_maxt, lvm_maxt)
    #boxplot(pvm, lvm, psubset, lsubset, pvm_maxt, lvm_maxt, 'SR')
    #boxplot(vm, pvm, hsubset, psubset, vm_maxt, pvm_maxt, var2='SR')
    vms = vm[16].ravel()[hsubset]
    pvms = pvm[16].ravel()[psubset]
    #lvms = lvm[16].ravel()[lsubset]
    regression(vms, pvms, record, 'HR', 'SR')
    #regression(lvms, [(a + b + c + d) / 4 for a, b, c, d in zip(pvms[::4], pvms[1::4], pvms[2::4], pvms[3::4])], record, 'LR', 'SR')
    #regression([(a + b + c + d) / 4 for a, b, c, d in zip(vms[::4], vms[1::4], vms[2::4], vms[3::4])], lvms, record, 'HR', 'LR')

    # plt.rcParams["figure.figsize"] = (14,8)
    # plt.plot(np.arange(0.01, 0.73, 0.01), vm_maxt, '--', color='blue', linewidth=0.5, label='HR')
    # plt.plot(np.arange(0.01, 0.73, 0.01), lvm_maxt, '--', color='red', linewidth=0.5, label='LR')
    # plt.plot(np.arange(0.01, 0.73, 0.01), pvm_maxt, color='black', label='SR')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.ylim([0, 2.5])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(pred_filepath[:-3] + '.png')
    # plt.clf()


def boxplot(vm, lvm, hsubset, lsubset, vm_maxt, lvm_maxt, var1='HR', var2='LR'):
    
    alphas = {'HR':0.0025, 'SR':0.0025, 'LR':0.25}
    colors = {'HR':'blue', 'SR':'black', 'LR':'red'}

    hn = sum(hsubset)
    ln = sum(lsubset)
    n = hn + ln 
    hr_lr = [var1]*hn + [var2]*ln
    dict_list = [{
        'Time (s)': [t/100]*n,
        'Data': hr_lr,
        'Velocity': [0.]*n
    } for t in range(1, 73)]

    for i, df_dict in enumerate(dict_list):
        v = vm[i].ravel()[hsubset]
        lv = lvm[i].ravel()[lsubset]
        df_dict['Velocity'][:hn] = v
        df_dict['Velocity'][hn:] = lv
    
    df = pd.concat([pd.DataFrame(df_dict) for df_dict in dict_list])

    sns.set(rc={'figure.figsize':(12,6)})
    sns.stripplot(y='Velocity', x='Time (s)', data=df[df.Data == var1], marker='o', alpha=alphas[var1], color=colors[var1])
    sns.stripplot(y='Velocity', x='Time (s)', data=df[df.Data == var2], marker='o', alpha=alphas[var2], color=colors[var2])
    plt.plot(np.arange(0, 72, 1), vm_maxt, '--', color=colors[var1], linewidth=1, label=var1)
    plt.plot(np.arange(0, 72, 1), lvm_maxt, '--', color=colors[var2], linewidth=1, label=var2)
    plt.ylim([0, 2.6])
    plt.title(pred_filepath[pred_filepath.rfind('/')+1:-3] + f' {var1} vs {var2}')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(pred_filepath[:-3] + f'_{var1}_{var2}.png')
    plt.clf()


def regression(vm, pvm, record, xvar, yvar):

    errorm = pvm - vm
    std = np.std(errorm)
    um = np.round(np.mean(errorm), 4)
    rmse = np.sqrt(np.mean((errorm)**2))
    
    record['RMSE'] = rmse
    record['se'] = std
    
    xhrsr = (vm+pvm)/2
    m, b = np.polyfit(vm, pvm, 1)
    top=np.round(1.96*std + um, 4)
    bot=np.round(-1.96*std + um, 4)
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(121), plt.scatter(vm, pvm, s=1.5, alpha=0.1, c='black')#, plt.title(titles[i//2])
    plt.subplot(121), plt.plot(vm, m*vm + b, linewidth=0.75, alpha=1)#, plt.title(titles[i//2])
    plt.subplot(121), plt.text(max(vm)//1.05, m*max(vm)+b,f'y={m:4.3f}x+{b:4.3f}')
    plt.subplot(121), plt.plot(vm, vm, linewidth=0.75, alpha=0.5)
    plt.subplot(121), plt.xlabel(f'{xvar} [m/s]')
    plt.subplot(121), plt.ylabel(f'{yvar} [m/s]')
    plt.subplot(121), plt.xlim([0, 2.5])
    plt.subplot(121), plt.ylim([0, 2.5])
    
    upper = 1.96*std+um
    lower = -1.96*std+um
    length = len(xhrsr)
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(122), plt.scatter(xhrsr, pvm-vm, s=1.5, alpha=0.1, c='black'), plt.title("")
    plt.subplot(122), plt.plot(xhrsr, np.repeat(upper, length), '--', color='green', linewidth=0.5, alpha=0.75)
    plt.subplot(122), plt.text(max(xhrsr), upper, f'{upper:4.3f}')
    plt.subplot(122), plt.plot(xhrsr, np.repeat(lower, length), '--', color='green', linewidth=0.5, alpha=0.75)
    plt.subplot(122), plt.text(max(xhrsr), lower, f'{lower:4.3f}')
    plt.subplot(122), plt.plot(xhrsr, np.repeat(um, length), '--', color='red', linewidth=0.5, alpha=0.75)
    plt.subplot(122), plt.text(max(xhrsr), um, f'{um:4.3f}')
    plt.subplot(122), plt.xlabel(f'0.5({yvar}+{xvar}) [m/s]')
    plt.subplot(122), plt.ylabel(f'{yvar}-{xvar} [m/s]')
    plt.subplot(122), plt.xlim([0, 2.5])
    plt.subplot(122), plt.ylim([-1, 1])
    plt.subplot(122), plt.yticks([-1, -0.5, 0, 0.5, 1])

    record['m'] = m
    record['b'] = b
    record['upper'] = top
    record['lower'] = bot
    # plt.tight_layout()
    plt.savefig(pred_filepath[:-3] + f'_RBA_{xvar}_{yvar}.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    
    downsample_factors = [4]
    all_networks = ['csp']
    noises = ['_NN']
    
    records = [{}] * (len(downsample_factors) * len(all_networks) * len(noises))
    idx = 0
    
    for downsample in downsample_factors:
        
        if downsample != 4:
            hr_filepath = f"data/trainG4HR_{downsample}x.h5"
        else:
            hr_filepath = "data/trainG4HR.h5"
        pred_folder = "data"
        
        for network in all_networks:
                       
            for i, nn in enumerate(noises):
                record = {}
                record['downsample'] = downsample
                record['network'] = network
                record['no noise'] = i
                fn = f"{network}_{downsample}x{nn}.h5" 
                lr_filepath = f"data/trainG4LR_4x{nn}.h5"
                with plt.style.context('bmh'):
                    pred_filepath = '{}/{}'.format(pred_folder, fn)
                    ic(hr_filepath, pred_filepath, lr_filepath)
                    plot_pressure(hr_filepath, pred_filepath, lr_filepath, downsample, record=record)
                
                records[idx] = record
                idx += 1 
    
    df = pd.DataFrame(records)
    df.to_csv(f'{pred_folder}/_metrics.csv')
    print(df)