'''Author: Liwen(Morris) Xu'''

import os
from os.path import join, splitext, exists
import argparse
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from vggish_input import wavfile_to_examples
from torch import hub

class VGGish_Pool(nn.Module):
    def __init__(self):
        super(VGGish_Pool, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
            
    def forward(self, x):
        x = self.features(x)
        return x

def main(model, audio_dir, save_dir, device, pool):
    audio_list = os.listdir(audio_dir)
    if not exists(save_dir):
        os.makedirs(save_dir)
    print('START EXTRACTION WITH TOTAL {} SAMPLES'.format(len(audio_list)))
    for i, file in enumerate(audio_list):
        audio_name = splitext(file)[0]
        if exists(join(audio_dir, file)) and exists(join(save_dir, audio_name + '.npy')):
            print('FOUND {}/{} EXTRACTED FEATURES'.format(i, len(audio_list)))
        else:
            audio = wavfile_to_examples(join(audio_dir, file)).to(device)
            np.save(join(save_dir, audio_name + '.npy'), model(audio).cpu().detach().numpy())
            if i%100:
                print('{}/{} FEATURES FINISHED EXTRACTION'.format(i, len(audio_list)))
    print('FINISHED ALL EXTRACTION!')
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='args of vggish_raw feature extraction')
    parser.add_argument('--audio_dir', type=str, help='dir where you save your original audio files')
    parser.add_argument('--save_dir', type=str, help='dir where you want to save the extracted features for embedding extraction')
    parser.add_argument('--device', type=int, help='GPU DEVICE INDEX that you are going to use')
    args = parser.parse_args()
    pool = mp.Pool() # Multithread processing
    
    
    # Firstly cut the last several FC layers in the original architecture and initialize the model 
    # with checkpoints released by Google.
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    vggish_pool = VGGish_Pool()
    vggish_pool_state = vggish_pool.state_dict()
    checkpoint = torch.hub.load('harritaylor/torchvggish', 'vggish')
    pool_state_dict = {k: v for k, v in checkpoint.state_dict().items() if k in vggish_pool_state}
    vggish_pool_state.update(pool_state_dict)
    vggish_pool.load_state_dict(vggish_pool_state)
    vggish_pool = vggish_pool.to(device)
    print('Model VGGish_Pool initialized successfully on {}'.format(device))
    
    # Start extration
    main(vggish_pool, args.audio_dir, args.save_dir, args.device, pool)
    
    
    
    
    
    
    

    