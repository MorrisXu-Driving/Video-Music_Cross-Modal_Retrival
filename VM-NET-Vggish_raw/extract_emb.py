#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
import scipy.io
from network_structure import Model_structure
from embedding import Embedding

'''the shape of music feature:(62, 512, 64, 4) only assume you extract music feature using  ../feature_extractor/torchvggish/torchvggish/extract_vgg_pool.py. You can use other feature extractor but 
you need to reshape every sample into (62, 512, 64, 4). Otherwise, it will not work.)

AND YOU NEED TO PUT THEM INTO A BATCH FORMAT YOURSELF IF YOU WANT TO ACCELERATE PROCESSING'''

# Create input sample
np.random.seed(0)
x.append(np.random.normal(size=[1024, 62, 512, 6, 4])) # music feature shape:(batch_size, time, feature_dim, height, width)
x.append(np.random.normal(size=[1024, 60, 2048])) # video feature shape:(batch, time, feature_dim)

# CHANGE YOUR OWN MODEL PATH IN embedding.py BEFORE starting extraction!!!!!!
embed_loader = Embedding()

# x is a list with x[0] refers to all music features and x[1] refers to all video features
embeddings = embed_loader(x)
print(embeddings[0].shape) # audio embedding
print(embeddings[1].shape) # video embedding

'''Correct Embedding output shapes should be both equal to (batch_size, 512)!!!'''



