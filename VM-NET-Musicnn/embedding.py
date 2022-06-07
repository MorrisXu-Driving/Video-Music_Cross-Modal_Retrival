#!/usr/bin/python
# -*- coding: utf-8 -*

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.compat.v1 import ConfigProto  
from tensorflow.compat.v1 import InteractiveSession  
config = ConfigProto()  
config.gpu_options.allow_growth = True  
session = InteractiveSession(config=config)  

import numpy as np
import os
from network_structure import Model_structure

# CHANGE YOUR OWN MODEL PATH CONFIGS BEFORE RUNNING!!!!!!
class Embedding(object):
    def __init__(self, model_class = Model_structure, model_path = "/PATH/TO/MODEL_SUMMARY_DIR"):
        self.batch_size = 1024
        self.model_path = model_path
        self.net = self.init_embed_model(model_class)
    
    def init_embed_model(self, model_class):
        self.net_opts = model_class.OPTS()
        self.net_opts.network_name = 'Wrapping Network'
        self.net_opts.x_dim = 500     
        self.net_opts.y_dim = 2048  # CHANGE THE VIDEO FEAT DIM IF YOU USE DIFFERENT MODEL FOR VISUAL FEATURE EXTARCTION
        self.net_opts.x_num_layer = 3
        self.net_opts.y_num_layer = 2
        self.net_opts.constraint_weights = [3, 1, 0.2, 0.2]
        self.net_opts.is_linear = False
        net = model_class(self.net_opts)
        net.construct()
        return net
    
    def __call__(self, input_feat):
        audio_feat = np.mean(input_feat[0], axis = 1, keepdims = False)
        video_feat = np.mean(input_feat[1], axis = 1)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            checkpoint_dir = os.path.join(self.model_path, 'checkpoints')
            step = 0
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                step = int(ckpt.model_checkpoint_path.split('-')[-1])
                step += 1
                print('Session restored successfully. step: {0}'.format(step))

            x_embed, y_embed = sess.run([self.net.x_embed, self.net.y_embed], feed_dict={
                self.net.x_data: audio_feat,
                self.net.y_data: video_feat,
                self.net.K: self.batch_size,
                self.net.keep_prob: 1., self.net.is_training: False
            }) 
        return x_embed, y_embed
                
                
