#!/usr/bin/python

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
from os.path import join
import scipy.io
import ipdb
from network_structure import Model_structure
from utils.Logger import Logger
from utils.dataiter import FeatLoader, GetBatch

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-4
                   , 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.9, 'Dropout keep probability rate.')
flags.DEFINE_integer('num_layer_x', 3, 'Constraint Weight xy')
flags.DEFINE_integer('num_layer_y',2, 'Constraint Weight yx')
flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_float('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_float('constraint_y', 0.2, 'Constraint Structure Weight y')
flags.DEFINE_integer('train_top_K', 1024, 'Top most K number for violation')
flags.DEFINE_integer('test_top_K', 1024, 'Top most K number for violation')
flags.DEFINE_float('weight_decay', 0, 'Weight decay.')

flags.DEFINE_integer('num_epochs', 202, 'number of epochs')
flags.DEFINE_integer('train_batch_size', 1024, 'Train Batch size.') 
flags.DEFINE_integer('validation_batch_size', 1024, 'Validation Batch size.')  
flags.DEFINE_integer('test_batch_size', 1024, 'Test batch size.')

# CHANGE YOUR OWN PATH CONFIGS BEFORE RUNNING!!!!!!
flags.DEFINE_string('train_data_dir', "/path/to/data_dir/train/", 'Directory to contain audio and rgb for training samples.')
flags.DEFINE_string('train_csv_path', "/path/to/train.csv", 'Path to the csv recording all training samples')
flags.DEFINE_string('test_data_dir', "/path/to/data_dir/test", 'Directory to contain audio and rgb for test samples.')
flags.DEFINE_string('test_csv_path', "/path/to/test.csv", 'Path to the csv recording all test samples')
flags.DEFINE_string('summaries_dir', "./models/MV_9k_efficient_b5_Avgpool_MUSICNN_penultimate_Structure_Nonlinear_single_loss_margin_0.5_emb_512_epochs_101_GlobalAvg", 'Directory to put the summary and log data.')

# You can change your own save_step based on the vacant space on your computer.
flags.DEFINE_integer('display_step', 10, 'Train display step.')
flags.DEFINE_integer('test_step', 10, 'Test step.')
flags.DEFINE_integer('save_step', 50, 'Checkpoint saving step.')

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


lr = tf.placeholder(tf.float32, name='learning_rate')
loss = net.loss
# loss = net.loss + net.dom_loss


global_step_ = tf.Variable(0, trainable=False)
decay_step = 1000

learning_rate_ = tf.train.exponential_decay(FLAGS.learning_rate, global_step_,decay_step, 0.96, staircase=True)
                                           

# learning_rate = FLAGS.learning_rate

train_top_k = FLAGS.train_top_K
test_top_k = FLAGS.test_top_K

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# Ensures that we execute the update_ops before performing the train_step (ref: http://ruishu.io/2016/12/27/batchnorm/)
with tf.control_dependencies(update_ops):
#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(loss, global_step=global_step_)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)

train_feats, _ = FeatLoader(FLAGS.train_csv_path, FLAGS.train_data_dir, FLAGS.train_batch_size)
num_batchs = int(train_feats[0].get_shape()[0].value/FLAGS.train_batch_size)
max_steps = num_batchs * FLAGS.num_epochs
print('Number of epochs {} and number of steps {} to train '.format(FLAGS.num_epochs, max_steps))

x_train_batch, y_train_batch, aff_train_xy = GetBatch(train_feats, FLAGS.num_epochs, FLAGS.train_batch_size, shuffle = False)
print('finished loading TRAIN samples and infering aff_train_xy')

test_feats, _ = FeatLoader(FLAGS.test_csv_path, FLAGS.test_data_dir, FLAGS.test_batch_size)
x_test_batch, y_test_batch, aff_test_xy = GetBatch(test_feats, FLAGS.num_epochs, FLAGS.test_batch_size, shuffle = False)
print('finished loading TEST samples and infering aff_test_xy')



with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
  
    checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
  
    logger = Logger(FLAGS.summaries_dir)
    logger.write(str(FLAGS.__flags))
    
  
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print('Session restored successfully. step: {0}'.format(step))
        step = step + 1

    max_step_acc_xy = [0, -5e3]
    max_step_acc_yx = [0, -5e3]
    
    try:
        for i in range(step, max_steps):
            p = float(i) / max_steps
            lamb = 0
            learning_rate = sess.run(learning_rate_)

            x_batch, y_batch, aff_xy = sess.run([x_train_batch, y_train_batch, aff_train_xy])
            sess.run(optimizer,feed_dict={
                              net.x_data:x_batch,
                              net.y_data:y_batch,
                              net.K:int(train_top_k),
                              net.aff_xy:aff_xy,
                              net.keep_prob:FLAGS.keep_prob,
                              lr: learning_rate, net.is_training: True, net.l: lamb})

            if (i+1) % FLAGS.display_step == 0:
                loss_cross_xy, loss_single_x, loss_cross_yx, loss_single_y, xy_rank_x, yx_rank_x, x_rank_x, xy_rank_y, yx_rank_y, y_rank_y, l,xy,yx,dl,da = sess.run([net.loss_cross_xy, net.loss_single_x, net.loss_cross_yx, net.loss_single_y, net.xy_rank_x, net.yx_rank_x, net.x_rank_x, net.xy_rank_y, net.yx_rank_y, net.y_rank_y, loss,net.recall_xy, net.recall_yx, net.dom_loss, net.dom_acc],feed_dict={
                          net.x_data:x_batch,
                          net.y_data:y_batch,
                          net.K:int(train_top_k),
                          net.aff_xy:aff_xy,
                          net.keep_prob:1., net.is_training: True, net.l: lamb})


                logger.write("[iter %d] (%s %s, %s %s)(%s %s, %s) (%s %s, %s) loss=%.4g, %s, %s "%(i+1, loss_cross_xy, loss_cross_yx, loss_single_x, loss_single_y,  xy_rank_x, yx_rank_x, x_rank_x, xy_rank_y, yx_rank_y, y_rank_y, l, xy, yx))
                short_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="triplet_loss", simple_value=float(l)),
                    tf.Summary.Value(tag="K", simple_value=float(train_top_k)),
                    tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
                    tf.Summary.Value(tag="dom_loss", simple_value=float(dl)),
                    tf.Summary.Value(tag="dom_acc", simple_value=float(da)),
                    tf.Summary.Value(tag="learning_rate", simple_value=float(learning_rate)),
                    tf.Summary.Value(tag="recall/xy_R@1", simple_value=float(xy[0])),
                    tf.Summary.Value(tag="recall/xy_R@5", simple_value=float(xy[1])),
                    tf.Summary.Value(tag="recall/xy_R@10", simple_value=float(xy[2])),
                    tf.Summary.Value(tag="recall/xy_R@20", simple_value=float(xy[3])),
                    tf.Summary.Value(tag="recall/xy_R@50", simple_value=float(xy[4])),
                    tf.Summary.Value(tag="recall/xy_R@100", simple_value=float(xy[5])),
                    tf.Summary.Value(tag="recall/yx_R@1", simple_value=float(yx[0])),
                    tf.Summary.Value(tag="recall/yx_R@5", simple_value=float(yx[1])),
                    tf.Summary.Value(tag="recall/yx_R@10", simple_value=float(yx[2])),
                    tf.Summary.Value(tag="recall/yx_R@20", simple_value=float(yx[3])),
                    tf.Summary.Value(tag="recall/yx_R@50", simple_value=float(yx[4])),
                    tf.Summary.Value(tag="recall/yx_R@100", simple_value=float(yx[5])),
                  ])
                train_writer.add_summary(short_summary, i)


            
            
            
            if (i+1) % FLAGS.test_step == 0:
                x_batch, y_batch, aff_xy =  sess.run([x_test_batch, y_test_batch, aff_test_xy])
                l,xy,yx, xy_idx, yx_idx,dl,da = sess.run([loss,net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx, net.dom_loss, net.dom_acc],feed_dict={      
                          net.x_data:x_batch,
                          net.y_data:y_batch,
                          net.K:int(test_top_k),  # caution: We currently use topK as test batch size
                          net.aff_xy:aff_xy,
                          net.keep_prob:1., net.is_training: False, net.l: lamb})  #actually, False


                logger.write("[TEST iter %d] loss=%.4g, %s, %s"%(i+1, l, xy, yx))
                short_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="triplet_loss", simple_value=float(l)),
                    tf.Summary.Value(tag="K", simple_value=float(test_top_k)),
                    tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
                    tf.Summary.Value(tag="dom_loss", simple_value=float(dl)),
                    tf.Summary.Value(tag="dom_acc", simple_value=float(da)),
                    tf.Summary.Value(tag="learning_rate", simple_value=float(learning_rate)),
                    tf.Summary.Value(tag="recall/xy_R@1", simple_value=float(xy[0])),
                    tf.Summary.Value(tag="recall/xy_R@5", simple_value=float(xy[1])),
                    tf.Summary.Value(tag="recall/xy_R@10", simple_value=float(xy[2])),
                    tf.Summary.Value(tag="recall/xy_R@20", simple_value=float(xy[3])),
                    tf.Summary.Value(tag="recall/xy_R@50", simple_value=float(xy[4])),
                    tf.Summary.Value(tag="recall/xy_R@100", simple_value=float(xy[5])),
                    tf.Summary.Value(tag="recall/yx_R@1", simple_value=float(yx[0])),
                    tf.Summary.Value(tag="recall/yx_R@5", simple_value=float(yx[1])),
                    tf.Summary.Value(tag="recall/yx_R@10", simple_value=float(yx[2])),
                    tf.Summary.Value(tag="recall/yx_R@20", simple_value=float(yx[3])),
                    tf.Summary.Value(tag="recall/yx_R@50", simple_value=float(yx[4])),
                    tf.Summary.Value(tag="recall/yx_R@100", simple_value=float(yx[5])),
                  ])

                test_writer.add_summary(short_summary, i)

            if (i+1) % FLAGS.save_step == 0:
                saver.save(sess,checkpoint_prefix,global_step=i)
                logger.write("[Checkpoint at step %d saved]"%(i+1))
                logger.write(FLAGS.summaries_dir)


    except tf.errors.OutOfRangeError:
        print("Done Training and Testing")
    finally:
        coord.request_stop()
        coord.join(threads)
        
        

                    
            

