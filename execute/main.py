import numpy as np
import parser
import os 
import tensorflow as tf
import time
import argparse
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir) 
from config.load import read_uea_dataset
import config.config as settings
import processing.process as pro
import modelib.capsule as cap
import modelib.capsule_loss as loss

devices = settings.devices
os.environ['CUDA_VISIBLE_DEVICES'] = devices



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

save_doc = settings.logs


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


total_iteration = total_size // batch_size
test_iteration = testx.shape[0] // test_size
X = tf.placeholder(tf.float32,shape=[None,length,dimension])
Y = tf.placeholder(tf.float32,shape=[None,num_classes])
dropout = tf.placeholder(tf.float32)
Is_train = tf.placeholder(tf.bool)



logit = cap.KCapMamba(X,num_classes,dropout,is_train=Is_train)

capmamba_loss = cap.margin_loss(Y,logit,name="logits")

capmamba_op = loss.Optimazer(capmamba_loss,choose=choose_option,lr=learning_rate,decay=decay_steps)


Teacheraccuracy = loss.Top_1_Accuracy(Y,logit)
model_size = cap.Totalcount()

start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    for i in range(epoch):
        for j in range(total_iteration):
            if j != total_iteration-1:
                batch_x = trainx[batch_size*j:batch_size*(j+1),...]
                batch_y = trainy[batch_size*j:batch_size*(j+1),...]
                _,losses = sess.run([capmamba_op,capmamba_loss],feed_dict={
                    X:batch_x, Y: batch_y,dropout:drop_rate, Is_train:True})
            else:
                batch_x = trainx[j*batch_size:,...]
                batch_y = trainy[j*batch_size:,...]
                _,losses = sess.run([capmamba_op,capmamba_loss],feed_dict={
                    X:batch_x, Y: batch_y,dropout:drop_rate, Is_train:True})
endTime = (time.time() - start_time)/60
print("%s--Training Time:%.6fmin"%(dataname,endTime))


