import tensorflow as tf 
import numpy as np 

eposilion = 1e-9

def kl_divergence_with_logits(p_logits, q_logits):
        p = p_logits
        log_p = tf.log(p_logits)
        log_q = tf.log(q_logits)
        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

def Top_1_Accuracy(label,predict):
    correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(predict,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
def CountTop1(label,predict):
    correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(predict,1))
    count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return count

def cross_entropy_loss(p,q):
    ce_loss =  tf.reduce_sum(- p*tf.log(q), -1)
    return ce_loss

def L1_loss(p,q):
    dif = tf.abs(p-q)
    l1_loss = tf.reduce_sum(dif,axis=-1)
    return l1_loss

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.reduce_sum(tf.select(condition, small_res, large_res),axis=-1)


def L2_loss(teacher,student):
    square =tf.square(teacher-student)
    square = tf.reduce_sum(square,axis=-1)
    return square


def DerectOptimzer(loss,choose=1,lr=0.0001):
    global_step =  tf.Variable(0, name='global_step')
    if choose == 1:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    elif choose == 2 :
        train_op = tf.train.RMSPropOptimizer(lr,decay=0.9,momentum=0.0).minimize(loss,global_step)
    elif choose == 3 :
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)
    else :
        train_op = None
    return train_op

def Optimazer(loss,choose=1,lr=0.001,decay=200):
    global_step =  tf.Variable(0, name='global_step')
    lr = tf.train.exponential_decay(lr, global_step, decay, 0.9, staircase=True)
    if choose == 1:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    elif choose == 2 :
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss,global_step)
    else:
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)
    return train_op

def Top_1_Accuracy(label,predict):
    correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(predict,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost

capsule_coefficient = 0.9
lamdaset = 0.5
def margin_loss(v_length,Y,name="Teacher"):
    with tf.variable_scope(name+"margin_loss"):
        max_l = tf.square(tf.maximum(0.0,capsule_coefficient-v_length))
        max_r = tf.square(tf.maximum(0.0,(v_length-(1-capsule_coefficient))))
        margin = tf.reduce_sum( tf.reduce_sum(Y * max_l  + lamdaset * (1-Y) * max_r,axis=-1))
        #margin = decay_weights(margin,weight_decay_rate)
        return margin



def BaseKL_withL2(teacher_store,student_store,label,T=1.0,core_theta=0.2):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = L2_loss(teacher_logit,student_logit)
    with tf.variable_scope("Total_loss"):
        total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss

def BaseKL_withL1(teacher_store,student_store,label,T=1.0,core_theta=0.2):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("L1_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = L1_loss(teacher_logit,student_logit)
    with tf.variable_scope("Total_loss"):
        total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss  

def BaseKL_withKL(teacher_store,student_store,label,T=1.0,core_theta=0.2):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_logit)
    with tf.variable_scope("Total_loss"):
        total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseKL_withResponseMutulKL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("MUtul_KL_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_logit)
        student_teacher_loss = kl_divergence_with_logits(tf.stop_gradient( student_logit),teacher_logit)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * teacher_student_loss + coe3 * student_teacher_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseKL_RelationSimple(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("MUtul_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * teacher_student_LRN_loss + coe3 * student_teacher_GRN_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 


def BaseKL_RelationSimpleCE(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("MUtul_CE_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = cross_entropy_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = cross_entropy_loss(teacher_GRN,student_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * teacher_student_LRN_loss + coe3 * student_teacher_GRN_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 


def BaseKL_RelationSimpleL2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Relation_KL_L2_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L2_loss(teacher_GRN,student_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * teacher_student_LRN_loss + coe3 * student_teacher_GRN_loss
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseKL_RelationSimpleMutuKL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Mutual_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
        student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+student_student_LRN_loss) + coe3 * (student_teacher_GRN_loss+teacher_student_GRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseKL_RelationSimpleResponse_KL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Response_KL_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
        student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_LRN)
        teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_GRN)
        teacher_student_loss = kl_divergence_with_logits( tf.stop_gradient(teacher_logit),student_logit )
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+student_student_LRN_loss) + coe3 * (student_teacher_GRN_loss+teacher_student_GRN_loss+teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseKL_RelationSimpleResponse_L2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Response_KL_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        student_student_LRN_loss = L2_loss(teacher_logit,student_LRN)
        teacher_student_GRN_loss = L2_loss(teacher_logit,student_GRN)
        teacher_student_loss = L2_loss( teacher_logit,student_logit )
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+student_student_LRN_loss) + coe3 * (student_teacher_GRN_loss+teacher_student_GRN_loss+teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

##### Deep Reslat
def BaseDeepRelation_KL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Mutual_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_A),student_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_B),student_LRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseDeepRelation_L2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Mutual_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L2_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L2_loss(Head_B,student_LRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseDeepRelation_CE(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Mutual_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = cross_entropy_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = cross_entropy_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  cross_entropy_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  cross_entropy_loss(Head_B,student_LRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def BaseDeepRelation_Huge(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Simple_Mutual_KL_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = huber_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = huber_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  huber_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  huber_loss(Head_B,student_LRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss

def BaseDeepRelation_L1(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Relation_L1_loss"):
        #teacher_logit = teacher_store["logit"] / T
        #student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L1_loss(teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L1_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L1_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L1_loss(Head_B,student_LRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

#####DeepResponsed
def DeepRelationResponse_KL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_KL_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_A),student_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_B),student_LRN)
        global_LRN_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_LRN)
        global_teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_logit)
        global_GRN_student_loss =  kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 


def DeepRelationResponse_L1(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_L1_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L1_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L1_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L1_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L1_loss(Head_B,student_LRN)
        global_LRN_student_loss = L1_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L1_loss(teacher_logit,student_logit)
        global_GRN_student_loss =  L1_loss( teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss

def DeepRelationResponse_CE(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = cross_entropy_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = cross_entropy_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  cross_entropy_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  cross_entropy_loss(Head_B,student_LRN)
        global_LRN_student_loss = cross_entropy_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = cross_entropy_loss(teacher_logit,student_logit)
        global_GRN_student_loss =  cross_entropy_loss( teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def EnsembleDeepRelationResponse_CE(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = cross_entropy_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = cross_entropy_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  cross_entropy_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  cross_entropy_loss(Head_B,student_LRN)
        global_LRN_student_loss = cross_entropy_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = cross_entropy_loss(teacher_logit,student_logit)
        global_GRN_student_loss =  cross_entropy_loss( teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss( student_store["logit"],label )
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * (teacher_loss+student_loss) + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 

def EnsembleDeepRelationResponse_KD(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_KD_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits( tf.stop_gradient(teacher_LRN),student_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient(teacher_GRN),student_GRN)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits( tf.stop_gradient(Head_A),student_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient(Head_B),student_LRN)
        global_LRN_student_loss = kl_divergence_with_logits(tf.stop_gradient(teacher_logit),student_LRN)
        global_teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient(teacher_logit),student_logit)
        global_GRN_student_loss =  kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss( student_store["logit"],label )
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * (teacher_loss+student_loss) + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 


def EnsembleDeepRelationResponse_L2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L2_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L2_loss(Head_B,student_LRN)
        global_LRN_student_loss = L2_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L2_loss(teacher_logit,student_logit)
        global_GRN_student_loss =  L2_loss( teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss( student_store["logit"],label )
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * (teacher_loss+student_loss) + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss 


def DeepRelationResponse_L2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_LRN,student_LRN)
        student_teacher_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L2_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L2_loss(Head_B,student_LRN)
        global_LRN_student_loss = L2_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L2_loss(teacher_logit,student_logit)
        global_GRN_student_loss =  L2_loss( teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss  

#############SelfDistillationLearning
def SelfLearningKD_L2(teacher_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        teacher_LRN = teacher_store["LRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_logit,Head_A)
        student_teacher_GRN_loss = L2_loss(teacher_logit,Head_B)
        head_a_teacher_LRN_loss  =  L2_loss( teacher_logit,teacher_LRN)
        head_b_teacher_LRN_loss  =  L2_loss(teacher_logit,teacher_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss

def SelfLearningKD_KL(teacher_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Deeep_Responsed_CE_loss"):
        teacher_logit = teacher_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        teacher_LRN = teacher_store["LRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient(  teacher_logit),Head_A)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),Head_B)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits( tf.stop_gradient( teacher_logit),teacher_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( teacher_logit),teacher_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
    with tf.variable_scope("Total_loss"):
        total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss)
        total_loss = tf.reduce_mean(total_loss)
        total_loss = decay_weights(total_loss,0.0005)
    return total_loss

############## MultualLearning

def BaseKL_Mutual_L2(teacher_store,student_store,label,T=1.0,core_theta=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = L2_loss(teacher_logit,student_logit)
    with tf.variable_scope("Total_loss"):
        #total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        teacher_total_loss = teacher_loss + core_theta * teacher_student_loss
        student_total_loss = student_loss + core_theta * teacher_student_loss
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

def BaseKL_Mutual_KL(teacher_store,student_store,label,T=1.0,core_theta=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient(teacher_logit),student_logit)
        student_teacher_loss = kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_logit)
    with tf.variable_scope("Total_loss"):
        #total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        teacher_total_loss = teacher_loss + core_theta * teacher_student_loss
        student_total_loss = student_loss + core_theta * student_teacher_loss
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss
#####################
def BaseKLDeep_Mutual_KL(teacher_store,student_store,label,T=1.0,core_theta=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN = kl_divergence_with_logits(tf.stop_gradient(teacher_LRN),student_LRN)
        student_teacher_LRN = kl_divergence_with_logits(tf.stop_gradient(student_LRN),teacher_LRN)
        teacher_student_GRN = kl_divergence_with_logits(tf.stop_gradient(teacher_GRN),student_GRN)
        student_teacher_GRN = kl_divergence_with_logits(tf.stop_gradient(student_GRN),teacher_GRN)
        teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient(teacher_logit),student_logit)
        student_teacher_loss = kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_logit)
    with tf.variable_scope("Total_loss"):
        #total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        teacher_total_loss = teacher_loss + core_theta * (teacher_student_loss+teacher_student_LRN+teacher_student_GRN)
        student_total_loss = student_loss + core_theta * (student_teacher_loss+ student_teacher_LRN + student_teacher_GRN)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

def BaseKLDeep_Mutual_L2(teacher_store,student_store,label,T=1.0,core_theta=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("L2_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN = L2_loss(teacher_LRN,student_LRN)
        student_teacher_LRN = L2_loss(student_LRN,teacher_LRN)
        teacher_student_GRN = L2_loss(teacher_GRN,student_GRN)
        student_teacher_GRN = L2_loss(student_GRN,teacher_GRN)
        teacher_student_loss = L2_loss(teacher_logit,student_logit)
        student_teacher_loss = L2_loss(student_logit,teacher_logit)
    with tf.variable_scope("Total_loss"):
        #total_loss = (1-core_theta) * teacher_loss + core_theta * teacher_student_loss
        teacher_total_loss = teacher_loss + core_theta * teacher_student_loss+teacher_student_LRN+teacher_student_GRN
        student_total_loss = student_loss + core_theta * student_teacher_loss+ student_teacher_LRN + student_teacher_GRN
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss



###Densely

def DenseMutualKD_KL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN) +   kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_GRN) +  kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_logit)
        teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN) +   kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_LRN) +  kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_logit)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_A),student_LRN) +   kl_divergence_with_logits(tf.stop_gradient( Head_A),student_GRN) +  kl_divergence_with_logits(tf.stop_gradient( Head_A),student_logit) + kl_divergence_with_logits(tf.stop_gradient( Head_A),student_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_B),student_LRN)+   kl_divergence_with_logits(tf.stop_gradient( Head_B),student_GRN) +  kl_divergence_with_logits(tf.stop_gradient( Head_B),student_logit) + kl_divergence_with_logits(tf.stop_gradient( Head_B),student_LRN)
        global_LRN_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_LRN)
        global_teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_logit)
        global_GRN_student_loss =  kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = kl_divergence_with_logits(tf.stop_gradient(student_LRN),Head_A) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),Head_B) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),teacher_LRN) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),teacher_GRN) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),teacher_logit)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient(student_GRN),teacher_GRN) + kl_divergence_with_logits(tf.stop_gradient(student_GRN),Head_A) + kl_divergence_with_logits(tf.stop_gradient(student_GRN),Head_B) + kl_divergence_with_logits(tf.stop_gradient(student_GRN),teacher_LRN) + kl_divergence_with_logits(tf.stop_gradient(student_GRN),teacher_logit)
        logit_student_loss = kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_logit)+kl_divergence_with_logits(tf.stop_gradient(student_logit),Head_A) + kl_divergence_with_logits(tf.stop_gradient(student_logit),Head_B) + kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_LRN) + kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss




###################Ours Model
def DeepMutualKD_KL(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_LRN),student_LRN)
        teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_GRN),student_GRN)
        head_a_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_A),student_LRN)
        head_b_teacher_LRN_loss  =  kl_divergence_with_logits(tf.stop_gradient( Head_B),student_LRN)
        global_LRN_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_LRN)
        global_teacher_student_loss = kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_logit)
        global_GRN_student_loss =  kl_divergence_with_logits(tf.stop_gradient( teacher_logit),student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = kl_divergence_with_logits(tf.stop_gradient(student_LRN),Head_A) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),Head_B) + kl_divergence_with_logits(tf.stop_gradient(student_LRN),teacher_LRN)
        student_teacher_GRN_loss = kl_divergence_with_logits(tf.stop_gradient(student_GRN),teacher_GRN)
        logit_student_loss = kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_logit)+kl_divergence_with_logits(tf.stop_gradient(student_logit),Head_A) + kl_divergence_with_logits(tf.stop_gradient(student_logit),Head_B) + kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_LRN) + kl_divergence_with_logits(tf.stop_gradient(student_logit),teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

def DeepMutualKD_L2(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_LRN,student_LRN)
        teacher_student_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L2_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L2_loss( Head_B,student_LRN)
        global_LRN_student_loss = L2_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L2_loss( teacher_logit,student_logit)
        global_GRN_student_loss =  L2_loss(teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = L2_loss(student_LRN,Head_A) + L2_loss(student_LRN,Head_B) + L2_loss(student_LRN,teacher_LRN)
        student_teacher_GRN_loss = L2_loss(student_GRN,teacher_GRN)
        logit_student_loss = L2_loss(student_logit,teacher_logit)+L2_loss(student_logit,Head_A) + L2_loss(student_logit,Head_B) + L2_loss(student_logit,teacher_LRN) + L2_loss(student_logit,teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

def DeepMutualKD_L1(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L1_loss( teacher_LRN,student_LRN)
        teacher_student_GRN_loss = L1_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L1_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L1_loss( Head_B,student_LRN)
        global_LRN_student_loss = L1_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L1_loss( teacher_logit,student_logit)
        global_GRN_student_loss =  L1_loss(teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = L1_loss(student_LRN,Head_A) + L1_loss(student_LRN,Head_B) + L1_loss(student_LRN,teacher_LRN)
        student_teacher_GRN_loss = L1_loss(student_GRN,teacher_GRN)
        logit_student_loss = L1_loss(student_logit,teacher_logit)+L1_loss(student_logit,Head_A) + L1_loss(student_logit,Head_B) + L1_loss(student_logit,teacher_LRN) + L1_loss(student_logit,teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss


def DeepMutualKD_CE(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = cross_entropy_loss( teacher_LRN,student_LRN)
        teacher_student_GRN_loss = cross_entropy_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  cross_entropy_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  cross_entropy_loss( Head_B,student_LRN)
        global_LRN_student_loss = cross_entropy_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = cross_entropy_loss( teacher_logit,student_logit)
        global_GRN_student_loss =  cross_entropy_loss(teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = cross_entropy_loss(student_LRN,Head_A) + cross_entropy_loss(student_LRN,Head_B) + cross_entropy_loss(student_LRN,teacher_LRN)
        student_teacher_GRN_loss = cross_entropy_loss(student_GRN,teacher_GRN)
        logit_student_loss = cross_entropy_loss(student_logit,teacher_logit)+cross_entropy_loss(student_logit,Head_A) + cross_entropy_loss(student_logit,Head_B) + cross_entropy_loss(student_logit,teacher_LRN) + cross_entropy_loss(student_logit,teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

def DeepMutualKD_Huber(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = huber_loss( teacher_LRN,student_LRN)
        teacher_student_GRN_loss = huber_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  huber_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  huber_loss( Head_B,student_LRN)
        global_LRN_student_loss = huber_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = huber_loss( teacher_logit,student_logit)
        global_GRN_student_loss =  huber_loss(teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = huber_loss(student_LRN,Head_A) + huber_loss(student_LRN,Head_B) + huber_loss(student_LRN,teacher_LRN)
        student_teacher_GRN_loss = huber_loss(student_GRN,teacher_GRN)
        logit_student_loss = huber_loss(student_logit,teacher_logit)+huber_loss(student_logit,Head_A) + huber_loss(student_logit,Head_B) + huber_loss(student_logit,teacher_LRN) + huber_loss(student_logit,teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss

######################################
########## For Test 
###########################################
def DeepMutualKD_L2Test(teacher_store,student_store,label,T=1.0,coe1=1.0,coe2=1.0,coe3=1.0):
    infor = {
        "T_LRN": None,
         "T_GRN":None,
         "S_LRN":None,
         "S_GRN":None,
    }
    with tf.variable_scope("Teacher_loss"):
        teacher_loss = margin_loss(  teacher_store["logit"],label)
    with tf.variable_scope("Student_loss"):
        student_loss = margin_loss(student_store["logit"],label)
    with tf.variable_scope("teacher_disttill_student_loss"):
        teacher_logit = teacher_store["logit"] / T
        student_logit = student_store["logit"] / T
        Head_A = teacher_store["HeadA"] / T
        Head_B = teacher_store["HeadB"] / T
        student_LRN = student_store["LRN"] / T
        teacher_LRN = teacher_store["LRN"] / T
        student_GRN = student_store["GRN"] / T
        teacher_GRN = teacher_store["GRN"] / T
        teacher_student_LRN_loss = L2_loss( teacher_LRN,student_LRN)
        teacher_student_GRN_loss = L2_loss(teacher_GRN,student_GRN)
        head_a_teacher_LRN_loss  =  L2_loss( Head_A,student_LRN)
        head_b_teacher_LRN_loss  =  L2_loss( Head_B,student_LRN)
        global_LRN_student_loss = L2_loss(teacher_logit,student_LRN)
        global_teacher_student_loss = L2_loss( teacher_logit,student_logit)
        global_GRN_student_loss =  L2_loss(teacher_logit,student_GRN)
        #student_student_LRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_LRN),teacher_LRN)
        #teacher_student_GRN_loss = kl_divergence_with_logits(tf.stop_gradient( student_GRN),teacher_GRN)
        teacher_total_loss = teacher_loss + teacher_student_LRN_loss +coe1* (teacher_student_GRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss) +coe3*(global_LRN_student_loss + global_teacher_student_loss +  global_GRN_student_loss)
        infor["T_LRN"] = teacher_student_LRN_loss + head_a_teacher_LRN_loss + head_b_teacher_LRN_loss
        infor["T_GRN"] = teacher_student_GRN_loss
    with tf.variable_scope("student_disttill_student"):
        student_teacher_LRN_loss = L2_loss(student_LRN,Head_A) + L2_loss(student_LRN,Head_B) + L2_loss(student_LRN,teacher_LRN)
        student_teacher_GRN_loss = L2_loss(student_GRN,teacher_GRN)
        logit_student_loss = L2_loss(student_logit,teacher_logit)+L2_loss(student_logit,Head_A) + L2_loss(student_logit,Head_B) + L2_loss(student_logit,teacher_LRN) + L2_loss(student_logit,teacher_GRN)
        student_total_loss = student_loss + coe1*student_teacher_LRN_loss+coe2*student_teacher_GRN_loss + coe3*logit_student_loss
        infor["S_LRN"] = student_teacher_LRN_loss
        infor["S_GRN"] = student_teacher_GRN_loss
    with tf.variable_scope("Total_loss"):
        #total_loss = coe1 * teacher_loss + coe2 * (teacher_student_LRN_loss+head_a_teacher_LRN_loss+global_LRN_student_loss) + coe3 * (student_teacher_GRN_loss+head_b_teacher_LRN_loss+global_GRN_student_loss+global_teacher_student_loss)
        #total_loss = tf.reduce_mean(total_loss)
        #total_loss = decay_weights(total_loss,0.0005)
        teacher_total_loss = tf.reduce_mean(teacher_total_loss)
        student_total_loss = tf.reduce_mean(student_total_loss)
        teacher_total_loss = decay_weights(teacher_total_loss,0.0005)
        student_total_loss = decay_weights(student_total_loss,0.0005)
    return teacher_total_loss,student_total_loss,infor