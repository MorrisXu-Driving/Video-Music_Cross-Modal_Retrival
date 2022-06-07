import numpy as np
import os
import multiprocessing as mp
from os.path import join, exists
from musicnn.extractor import extractor

def main(audio_dir, save_dir, pool):
    file_list = os.listdir(audio_dir)
    for i, file in enumerate(file_list):
        vid = os.path.splitext(file)[0]
        if exists(join(save_dir, 'penultimate', vid + '.npy')):
            if i % 50 == 0:
                print('Extracting {}/{} features exists'.format(i, len(file_list)))
        else:
            _, _, features = extractor(join(wav_dir, file), model='MSD_musicnn_big', extract_features = True)
            np.save(join(save_dir, 'penultimate', vid + '.npy'), features['penultimate'])
            if i % 50 == 0:
                print('Finished Extracting {}/{} features'.format(i, len(file_list)))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='args of musicnn feature extraction')
    parser.add_argument('--audio_dir', type=str, help='dir where you save your original audio files')
    parser.add_argument('--save_dir', type=str, help='dir where you want to save the extracted features for embedding extraction')
    args = parser.parse_args()
    pool = mp.Pool()
    
    
    main(args.audio_dir, args.save_dir, pool)
    print('Finished Extracting all features using MSD_musicnn_big, total {} features'.format(len(file_list)))

        
                                                                  
    