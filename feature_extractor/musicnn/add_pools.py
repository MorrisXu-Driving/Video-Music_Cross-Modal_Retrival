import numpy as np
import pandas as pd
import os
from os.path import join, exists
'''
In script aims to add features from max_pool and mean_pool in the middle end of Musicnn. To use this script:

1. Arrange your feature dir like below:
├── musicnn_feature
    ├── max_pool
        └──features of audio stored in id.npy format 
    └── mean_pool
        └──features of video stored in id.npy format 
        
2. Change the feat_dir to the dir you place musicnn_feat/

3. Run this script.
'''

if __name__ == '__main__':
    feat_dir = 'PATH TO Musicnn features'
    max_dir = join(feat_dir,  'max_pool')
    mean_dir = join(feat_dir,  'mean_pool')
    mean_max_add_dir = join(feat_dir,  'mean_max_add')
    max_list, mean_list = os.listdir(max_dir), os.listdir(mean_dir)
    assert max_list == mean_list
    for i, file in enumerate(max_list):
        if exists(join(max_dir, file)) and exists(join(mean_dir, file)):
            max_pool = np.load(join(max_dir, file))
            mean_pool = np.load(join(mean_dir, file))
            mean_max_add = np.add(max_pool, mean_pool)
            np.save(join(mean_max_add_dir, file), mean_max_add)
        else:
            if not exists(join(max_dir, file)):
                raise ValueError("File {} absent".format(join(max_dir, file)))
            elif not exists(join(mean_dir, file)):
                raise ValueError("File {} absent".format(join(mean_dir, file)))
        if i % 100 == 0:
            print('finished computing {}/{} mean_max_add features'.format(i, len(max_list)))
    print('mean_max_add pool extraction finished!')
                
        

    
