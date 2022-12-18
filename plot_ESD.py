import torch
import weightwatcher as ww
import os
from multiprocessing import Pool
import cv2

ckpt_path = '/GPFS/data/ziwang/projects/sentiment/ckpts/tmp_same_dim_linear'
save_figs_path = '/GPFS/data/ziwang/projects/sentiment/save_figs/tmp_same_dim_linear'

def rearange_figs(i):
    model_path = ckpt_path + '/epoch_{}.pt'.format(i)
    model = torch.load(model_path)

    watcher = ww.WeightWatcher(model=model)
    save_path = save_figs_path + '/ori_figs/epoch_{}/'.format(i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    details = watcher.analyze(vectors=False, fit='PL', plot = True, savefig=save_path)
    # print(details)

    res = []
    for root,directory,files in os.walk(save_path):
        for filename in files:
            if 'esd.png' in filename:
                res.append(filename)
                split_result = filename.split('.')
                new_file_path = os.path.join(save_figs_path + '/esd', split_result[1])
                if not os.path.exists(new_file_path):
                    os.makedirs(new_file_path)
                    
                img = cv2.imread(os.path.join(save_path, filename))
                cv2.imwrite(os.path.join(new_file_path, 'epoch_{}.png'.format(i)), img)
                
    for root,directory,files in os.walk(save_path):
        for filename in files:
            if 'esd2.png' in filename:
                res.append(filename)
                split_result = filename.split('.')
                new_file_path = os.path.join(save_figs_path + '/esd2', split_result[1])
                if not os.path.exists(new_file_path):
                    os.makedirs(new_file_path)
                    
                img = cv2.imread(os.path.join(save_path, filename))
                cv2.imwrite(os.path.join(new_file_path, 'epoch_{}.png'.format(i)), img)
    

if __name__ == '__main__':
    with Pool(10) as p:
        p.map(rearange_figs, list(range(200)))