#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
import scipy.io
from utils.dataiter import FeatLoader, GetBatch
from network_structure import Model_structure

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_layer_x', 3, 'Constraint Weight xy')
flags.DEFINE_integer('num_layer_y',2, 'Constraint Weight yx')
flags.DEFINE_integer('test_batch_size', 1024, 'Test batch size.') #flags.DEFINE_integer('test_batch_size', 1000, 'Test batch size.')

# CHANGE YOUR OWN PATH CONFIGS BEFORE RUNNING!!!!!!
flags.DEFINE_string('test_data_dir', "path/to/data_dir/test", 'Directory to contain audio and rgb for test samples.')
flags.DEFINE_string('test_csv_path', "/path/to/test.csv", 'Path to the csv recording all test samples')
flags.DEFINE_string('summaries_dir', "./models/MV_9k_efficient_b5_Avgpool_MUSICNN_penultimate_Structure_Nonlinear_single_loss_margin_0.5_emb_512_epochs_101_GlobalAvg", 'Directory to put the summary and log data.')

flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_float('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_float('constraint_y', 0.2, 'Constraint Structure Weight y')

net_opts = Model_structure.OPTS()
net_opts.network_name = 'Wrapping Network'
net_opts.x_dim = 500
net_opts.y_dim = 2048  # CHANGE THE VIDEO FEAT DIM IF YOU USE DIFFERENT MODEL FOR VISUAL FEATURE EXTARCTION
net_opts.x_num_layer = FLAGS.num_layer_x
net_opts.y_num_layer = FLAGS.num_layer_y
net_opts.constraint_weights = [FLAGS.constraint_xy, FLAGS.constraint_yx, FLAGS.constraint_x, FLAGS.constraint_y]
net_opts.is_linear = False
net = Model_structure(net_opts)
net.construct()

test_feats, _ = FeatLoader(FLAGS.test_csv_path, FLAGS.test_data_dir, FLAGS.test_batch_size)
x_test_batch, y_test_batch, aff_test_xy = GetBatch(test_feats, 10, FLAGS.test_batch_size, shuffle = False)
print('finished loading TEST samples and infering aff_test_xy')
batch_num = test_feats[0].shape[0]//FLAGS.test_batch_size

Recall_xy, Recall_yx = [], []

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    lamb = 0
    K = 100
    
    checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.all_model_checkpoint_paths:
        for path in ckpt.all_model_checkpoint_paths:
            saver.restore(sess, path)
            step = path.split('/')[-1].split('-')[-1]
            print('Session restored successfully. step: {0}'.format(step))
            for i in range(batch_num):
                x_batch, y_batch, aff_xy =  sess.run([x_test_batch, y_test_batch, aff_test_xy])
                xy, yx, xy_idx, yx_idx = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx], feed_dict={
                    net.x_data: x_batch,
                    net.y_data: y_batch,
                    net.K: K,
                    net.aff_xy: aff_xy,
                    net.keep_prob: 1., net.is_training: False})  # actually, False
                print("[batch %d] xy: %s, yx: %s, " % (i, xy, yx))
                Recall_xy.append(np.asarray(xy, dtype = float))
                Recall_yx.append(np.asarray(yx, dtype = float))
                print("=" * 130)
            overall_xy_recall = np.mean(np.stack(Recall_xy), axis = 0)
            overall_yx_recall = np.mean(np.stack(Recall_yx), axis = 0)
            print("Overall xy R@1={}, R@5={}, R@10={}, R@20={}, R@50={}, R@100={}".format(overall_xy_recall[0], overall_xy_recall[1], overall_xy_recall[2], overall_xy_recall[3], overall_xy_recall[4], overall_xy_recall[5]))
            print("Overall yx R@1={}, R@5={}, R@10={}, R@20={}, R@50={}, R@100={}".format(overall_yx_recall[0], overall_yx_recall[1], overall_yx_recall[2], overall_yx_recall[3], overall_yx_recall[4], overall_yx_recall[5]))

    coord.request_stop()
    coord.join(threads)

