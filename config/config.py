import os

load_root = "../data/"
dataname = "AtrialFibrillation"
fill_strategy =  'column_mean' 
devices = '0'
epochs = 1

options = 1
learning_rate = 0.001
decay_steps = 100
drop_rate = 0.8

batch_num = 1 ##### batch_size = total_train_num // batch_num

test_num = 1 #### test_batch = total_test_num // batch_num

# Filling strategy: column_mean(column mean)/sample_mean(sample mean)
logs = "../data/log/"
