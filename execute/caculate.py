import parser
import os 
import tensorflow as tf
import time
import numpy as np
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir) 
from config.load import read_uea_dataset
import config.config as settings
import processing.process as pro
import modelib.capsule as cap
import modelib.capsule_loss as loss

devices = settings.devices
os.environ['CUDA_VISIBLE_DEVICES'] = devices



def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1e6, params.total_parameters/1e6))
    infor = 'FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1e6, params.total_parameters/1e6)
    return infor


save_doc = settings.logs

batch_size = settings.batch_num
test_size = settings.test_num
epoch = settings.epochs
choose_option = settings.options
learning_rate = settings.learning_rate
decay_steps = settings.decay_steps
drop_rate = settings.drop_rate

load_root = settings.load_root
dataname = settings.dataname
fill_strategy = settings.fill_strategy
# Filling strategy: column_mean(column mean)/sample_mean(sample mean)

datainfor = read_uea_dataset(dataname, root_dir=load_root)



trainx = datainfor["TrainX"]
trainy = datainfor["TrainY"]
testx = datainfor["TestX"]
testy = datainfor["TestY"]


trainx = pro.NormalizationFeatures(trainx)
testx = pro.NormalizationFeatures(testx)

num_classes = len(np.unique(testy))
length = trainx.shape[1]
dimension =  trainx.shape[2]

trainy = pro.OneHot(trainy,num_classes)
testy = pro.OneHot(testy,num_classes)

total_size = trainx.shape[0] 

file_name = os.path.join(save_doc,"Complexity.txt")
f = open(file_name,"a+")
with tf.Graph().as_default() as graph:
    X = tf.placeholder(tf.float32,shape=[None,length,dimension])
    Y = tf.placeholder(tf.float32,shape=[None,num_classes])
    dropout = tf.placeholder(tf.float32)
    Is_train = tf.placeholder(tf.bool)
    logit = cap.KCapMamba(X,num_classes,0.5,is_train=Is_train)
    infor = stats_graph(graph)

f.writelines("KCapMamba----name:"+name+"======Information:"+infor+"\n")
f.close()
