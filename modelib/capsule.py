import scipy
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
def Totalcount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
        # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("The Total params:",total_parameters/1e6,"M")
    #print("The Total params:",total_parameters/1e6,"M")
    return total_parameters/1e6


eposilion = 1e-9

def batch_normal(value,is_training=False,name='batch_norm'):
    if is_training is True:
         return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
def Active(x,mode='relu'):
    if mode == 'relu' :
        return tf.nn.relu(x)
    elif mode == 'leaky_relu' :
        return tf.nn.leaky_relu(x,alpha=0.1)
    else:
        return tf.nn.tanh(x)
mode = 'leaky_relu'
def CNNs(value,channel,kernel,stride,padding,is_training):
    conv = tf.contrib.layers.conv1d(value,channel,kernel,stride=stride,padding="SAME",activation_fn=None)
    bn = batch_normal(conv,is_training)
    res = Active(bn,mode)
    return res


def SquashCNNs(value,channel,kernel,stride,padding,is_training):
    conv = tf.contrib.layers.conv1d(value,channel,kernel,stride=stride,padding="SAME",activation_fn=None)
    bn = batch_normal(conv,is_training)
    res = squash(bn)
    return res

def squash(value):
    vec_square_norm = tf.reduce_sum(tf.square(value),-2,keepdims=True )
    scales = vec_square_norm / (1+ vec_square_norm) / tf.sqrt(vec_square_norm + eposilion)
    return value * scales


def routing(input, b_IJ, num_outputs=6, num_dims=8):
    ''' The routing algorithm.

    Args:
        input:  shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    input_shape = get_shape(input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape()

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() 

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    n = 3  #3
    # line 3,for r iterations do
    for r_iter in range(n):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == n-1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() 

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape()
            elif r_iter < n-1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from
                # then matmul 
                # batch_size dim, resulting
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)



def Init_selfCapsuleAttention(X,channel,nums):
      channel = nums * channel
      with tf.variable_scope("capsule_query"):
            query = slim.conv1d(X,channel,1,stride=1,padding="SAME",activation_fn=None)
      with tf.variable_scope("capsule_key"):
            key =  slim.conv1d(X,channel,1,stride=1,padding="SAME",activation_fn=None)
      with tf.variable_scope("capsule_values"):
            value =  slim.conv1d(X,channel,1,stride=1,padding="SAME",activation_fn=None)
      with tf.variable_scope("capsule_Attended"):
             _,b,c = get_shape(key)
             key = tf.reshape(key,[-1,c,b])
             attend = tf.matmul(query,key)
             attend = tf.nn.softmax(attend,-1)
             attend = tf.sqrt(reduce_sum(tf.square(attend),axis=2,keepdims=True)+eposilion)
             pred = attend * value
      out = slim.conv1d(X,channel,1,1,"SAME",activation_fn=None)
      para = 1.0
      result = out * para+ pred
      return result

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def AttentionCapsuleScaleDot(Q,K,V,key_masks,dropout=0.0,training=True,scope="scaled_capsule_attention"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        attend = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        attend /= d_k ** 0.5
        attend = mask(attend,key_masks,type="f")
        #attend = tf.layers.dropout(attend,rate=dropout,training=training)
        #attend = tf.nn.softmax(attend,-1)
        attend = tf.sqrt(reduce_sum(tf.square(attend),axis=2,keepdims=True)+eposilion)
        pred = attend * V
        return pred





def self_attention_layer(X, C1, key_masks=None, dropout=0.0, training=True, 
                        scope="self_attention", initializer=tf.glorot_uniform_initializer(),
                        use_residual=True, activation=None):
    """
    Self-attention mechanism layer
    
    Parameters:
        X: Input tensor with shape [B, L, C]
        C1: Output dimension
        key_masks: Mask tensor with shape [B, L] or [B, 1, L]
        dropout: Dropout rate
        training: Whether in training mode
        scope: Variable scope
        initializer: Weight initializer
        use_residual: Whether to use residual connection
        activation: Output activation function
    
    Returns:
        Output tensor with shape [B, L, C1]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Get input dimensions
        B = tf.shape(X)[0]
        L = tf.shape(X)[1]
        C = X.get_shape().as_list()[2]
        
        # Create all-ones mask if not provided
        if key_masks is None:
            key_masks = tf.ones([B, L], dtype=tf.float32)
        
        # Adjust mask shape to [B, 1, L]
        if len(key_masks.get_shape()) == 2:
            key_masks = tf.expand_dims(key_masks, axis=1)
        
        # Linear transformation to generate Q, K, V
        # Project to the same dimension (can be C1 or other dimensions)
        d_model = C1  # Can be adjusted to other dimensions
        
        Q = tf.layers.dense(X, d_model, 
                           kernel_initializer=initializer,
                           name="query")
        K = tf.layers.dense(X, d_model, 
                           kernel_initializer=initializer,
                           name="key")
        V = tf.layers.dense(X, C1, 
                           kernel_initializer=initializer,
                           name="value")
        
        # Apply dropout
        Q = tf.layers.dropout(Q, rate=dropout, training=training)
        K = tf.layers.dropout(K, rate=dropout, training=training)
        V = tf.layers.dropout(V, rate=dropout, training=training)
        
        # Apply capsule scaled dot-product attention
        attention_output = AttentionCapsuleScaleDot(Q, K, V, key_masks, 
                                                   dropout=dropout, 
                                                   training=training)
        
        # Output projection
        output = tf.layers.dense(attention_output, C1, 
                                kernel_initializer=initializer,
                                name="output_projection")
        
        # Apply dropout
        output = tf.layers.dropout(output, rate=dropout, training=training)
        
        # Residual connection
        if use_residual:
            # Linear transformation needed if input and output dimensions differ
            if C != C1:
                X_proj = tf.layers.dense(X, C1, 
                                        kernel_initializer=initializer,
                                        name="residual_projection")
                output = output + X_proj
            else:
                output = output + X
        
        # Layer normalization
        output = tf.contrib.layers.layer_norm(output, begin_norm_axis=-1)
        
        # Activation function
        if activation is not None:
            output = activation(output)
        
        return output










def MultiHeadCapsuleAttention(queries,keys,values,key_masks,d_model=64,num_heads=8,dropout=0.0,training=True,scope="Multiattention"):
    #d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries,d_model,use_bias=True)
        K = tf.layers.dense(keys,d_model,use_bias=True)
        V = tf.layers.dense(values,d_model,use_bias=True)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        outputs = AttentionCapsuleScaleDot(Q_,K_,V_,key_masks,dropout,training)
        outputs =tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
        outputs += tf.layers.dense(queries,d_model)
        outputs = ln(outputs)
        return outputs


def mask(inputs,key_masks=None,type=None):
    padding_num = -2 ** 32 +1
    if type in ("k","key","keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks,[tf.shape(inputs)[0]//tf.shape(key_masks)[0],1])
        key_masks = tf.expand_dims(key_masks,1)
        output = inputs + key_masks * padding_num

    elif type in ("f","future","right"):
        diag_vals = tf.ones_like(inputs[0,:,:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs)[0],1,1])
        paddings = tf.ones_like( future_masks ) * padding_num
        output = tf.where(tf.equal(future_masks,0),paddings,inputs)
    else:
        print("Check if you entered type correctly")

    return output

def Dense(inputs,units):
    x = tf.layers.dense(inputs,units,use_bias=True)
    x = ln(x)
    return x

def BaseTransformer(inputs,key_masks,d_model=64,num_heads=8,dropout=0.0,training=True,scope="BaseTransformer"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        multihead = MultiHeadCapsuleAttention(inputs,inputs,inputs,key_masks,d_model,num_heads,dropout,training)
        x1 = Dense(squash(multihead),d_model)
        x2 = Dense(squash(x1),d_model)
        outputs = squash( x2+ multihead  )
        return outputs

def PrimaryTimeS(vector,kernel,channel,stride,is_train,reshapes):
    values = CNNs(vector,channel,kernel,stride,"SAME",is_train)
    values = tf.reshape(values,reshapes)
    capsu = squash(values)
    return capsu

def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost

def getNextStride(step):
    if step %2 == 0:
        return step//2
    else :
        return step//2 +1
def DiscountStep(step,index):
    for i in range(index):
        step = getNextStride(step)
    return step

Teacher_transformer_block_layers = 2
Teacher_transformer_perlayer = 48
Teacher_transformer_head = 8

def CapsuleTransformerTeacher(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,128,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,24)
        with tf.variable_scope("second_headB"):
            sec_head = SquashCNNs(X,7,128,2,padding,is_train)
            sec_primary = PrimaryTimeS(sec_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            sec_fc_function = tf.reshape(sec_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            sec_target_shape = get_shape(sec_primary)
            sec_blu = tf.zeros([sec_target_shape[0],sec_target_shape[1],num_classes,1,1])
            sec_caps = routing(sec_fc_function,sec_blu,num_classes,24)

    with tf.variable_scope("transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Teacher_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
        with tf.variable_scope("transformer_routing"):
            trans_routing_head = 8
            trans_capsule_shape = get_shape(trans_capsules)
            trans_fc_function = tf.reshape(trans_capsules,shape=(-1,int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),1,trans_routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=trans_routing_head*2)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([top_caps,sec_caps,trans_caps],axis=3)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length

def CpasulePrediction(caps,num_classes):
    with tf.variable_scope("predictions"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
        return v_length


def CapsuleTransformerTeacherTraining(X,num_classes,length,is_train=True):
    padding = "SAME"
    store = {
        "HeadA":None,
        "HeadB":None,
        "LRN":None,
         "GRN":None,
         "logit":None,
    }
    with tf.variable_scope("local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,128,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,24)
            store ["HeadA"] = CpasulePrediction(top_caps,num_classes)
        with tf.variable_scope("second_headB"):
            sec_head = SquashCNNs(X,7,128,2,padding,is_train)
            sec_primary = PrimaryTimeS(sec_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            sec_fc_function = tf.reshape(sec_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            sec_target_shape = get_shape(sec_primary)
            sec_blu = tf.zeros([sec_target_shape[0],sec_target_shape[1],num_classes,1,1])
            sec_caps = routing(sec_fc_function,sec_blu,num_classes,24)
            store["HeadB"] = CpasulePrediction(sec_caps,num_classes)
            store["LRN"] =  CpasulePrediction(tf.concat([top_caps,sec_caps],axis=3),num_classes)
    with tf.variable_scope("transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Teacher_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
        with tf.variable_scope("transformer_routing"):
            trans_routing_head = 8
            trans_capsule_shape = get_shape(trans_capsules)
            trans_fc_function = tf.reshape(trans_capsules,shape=(-1,int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),1,trans_routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=trans_routing_head*2)
            store["GRN"] = CpasulePrediction(trans_caps,num_classes)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([top_caps,sec_caps,trans_caps],axis=3)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
        store["logit"] = v_length
    return store



def CapsuleLocalTeacher(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,128,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,24)
        with tf.variable_scope("second_headB"):
            sec_head = SquashCNNs(X,7,128,2,padding,is_train)
            sec_primary = PrimaryTimeS(sec_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            sec_fc_function = tf.reshape(sec_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            sec_target_shape = get_shape(sec_primary)
            sec_blu = tf.zeros([sec_target_shape[0],sec_target_shape[1],num_classes,1,1])
            sec_caps = routing(sec_fc_function,sec_blu,num_classes,24)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([top_caps,sec_caps],axis=3)
        caps = tf.layers.dropout(caps,rate=0.5,training=is_train)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length


def CapsuleGlobalTransformerTeacher(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Teacher_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
        with tf.variable_scope("transformer_routing"):
            trans_routing_head = 8
            trans_capsule_shape = get_shape(trans_capsules)
            trans_fc_function = tf.reshape(trans_capsules,shape=(-1,int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),1,trans_routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=trans_routing_head*2)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = trans_caps
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length

def CapsuleGlobalTransformerTeacherWithoutRouting(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Teacher_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.layers.average_pooling1d(trans_capsules,length,1)
    with tf.variable_scope("Prediction"):
        #caps = tf.squeeze(caps,axis=1)num
        caps = tf.layers.dense(caps,num_classes)
        #print(caps)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=1,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length

def CapsuleLocalTeacherA(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,128,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,24)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([top_caps],axis=3)
        caps = tf.layers.dropout(caps,rate=0.5,training=is_train)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length

def CapsuleLocalTeacherB(X,num_classes,length,is_train=True):
    padding = "SAME"
    with tf.variable_scope("local_feature_relationships"):
        with tf.variable_scope("second_headB"):
            sec_head = SquashCNNs(X,7,128,2,padding,is_train)
            sec_primary = PrimaryTimeS(sec_head,9,128,2,is_train,[-1,DiscountStep(length,2)*16,8,1])
            sec_fc_function = tf.reshape(sec_primary,shape=(-1,DiscountStep(length,2)*16,1, 8,1))
            sec_target_shape = get_shape(sec_primary)
            sec_blu = tf.zeros([sec_target_shape[0],sec_target_shape[1],num_classes,1,1])
            sec_caps = routing(sec_fc_function,sec_blu,num_classes,24)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([sec_caps],axis=3)
        caps = tf.layers.dropout(caps,rate=0.5,training=is_train)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length




def CapsuleTransformerStudent(X,num_classes,length,is_train=True):
    Student_transformer_block_layers = 2
    Teacher_transformer_perlayer = 24
    Teacher_transformer_head = 8
    padding = "SAME"
    with tf.variable_scope("Student_local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,64,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,64,2,is_train,[-1,DiscountStep(length,2)*8,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*8,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,12)
    with tf.variable_scope("Student_transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Student_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
        with tf.variable_scope("transformer_routing"):
            trans_routing_head = 8
            trans_capsule_shape = get_shape(trans_capsules)
            trans_fc_function = tf.reshape(trans_capsules,shape=(-1,int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),1,trans_routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(Teacher_transformer_perlayer*Student_transformer_block_layers//trans_routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=trans_routing_head*2)
    with tf.variable_scope("Student_ConcatenatedLayer"):
        #print(top_caps,trans_capsules)
        caps = tf.concat([top_caps,trans_caps],axis=3)
    with tf.variable_scope("Student_Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length



def CapsuleTransformerStudentTraining(X,num_classes,length,is_train=True):
    Student_transformer_block_layers = 2
    Teacher_transformer_perlayer = 24
    Teacher_transformer_head = 8
    padding = "SAME"
    store = {
        "LRN":None,
         "GRN":None,
         "logit":None,
    }
    with tf.variable_scope("Student_local_feature_relationships"):
        with tf.variable_scope("first_headA"):
            top_head = SquashCNNs(X,9,64,2,padding,is_train)
            top_primary = PrimaryTimeS(top_head,9,64,2,is_train,[-1,DiscountStep(length,2)*8,8,1])
            top_fc_function = tf.reshape(top_primary,shape=(-1,DiscountStep(length,2)*8,1, 8,1))
            top_target_shape = get_shape(top_primary)
            top_blu = tf.zeros([top_target_shape[0],top_target_shape[1],num_classes,1,1])
            top_caps = routing(top_fc_function,top_blu,num_classes,12)
            store["LRN"] = CpasulePrediction(top_caps,num_classes)
    with tf.variable_scope("Student_transformerCapsule"):
        trans_capsules = X
        trans_mask = tf.math.equal(X,0)
        with tf.variable_scope("transformers"):
            for i in range(Student_transformer_block_layers):
                with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                    trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                     num_heads=Teacher_transformer_head,dropout=0.2,training=is_train)
        with tf.variable_scope("transformer_routing"):
            trans_routing_head = 8
            trans_capsule_shape = get_shape(trans_capsules)
            trans_fc_function = tf.reshape(trans_capsules,shape=(-1,int(Teacher_transformer_perlayer*Teacher_transformer_block_layers//trans_routing_head),1,trans_routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(Teacher_transformer_perlayer*Student_transformer_block_layers//trans_routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=trans_routing_head*2)
            store["GRN"] = CpasulePrediction(trans_caps,num_classes)
    with tf.variable_scope("Student_ConcatenatedLayer"):
        #print(top_caps,trans_capsules)
        caps = tf.concat([top_caps,trans_caps],axis=3)
    with tf.variable_scope("Student_Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
        store["logit"] = v_length
    return store


def ssm_layer_simple(X, C2, state_dim=None, trainable=True, 
                    kernel_initializer=tf.glorot_uniform_initializer(), 
                    bias_initializer=tf.zeros_initializer(),
                    activation=None,
                    return_states=False):
    """
    Simplified version of SSM layer, suitable for static shape inputs
    """
    if state_dim is None:
        state_dim = C2
    
    # Get input dimensions
    input_shape = X.get_shape().as_list()
    C1 = input_shape[2]
    
    # Initializer handling
    if isinstance(kernel_initializer, str):
        kernel_initializer = tf.glorot_uniform_initializer()
    
    # Parameter initialization
    A = tf.get_variable('A', [state_dim, state_dim], initializer=kernel_initializer, trainable=trainable)
    B = tf.get_variable('B', [state_dim, C1], initializer=kernel_initializer, trainable=trainable)
    C = tf.get_variable('C', [C2, state_dim], initializer=kernel_initializer, trainable=trainable)
    D = tf.get_variable('D', [C2, C1], initializer=kernel_initializer, trainable=trainable)
    
    # Initial state
    batch_size = tf.shape(X)[0]
    initial_state = tf.zeros([batch_size, state_dim], dtype=X.dtype)
    
    # Unstack sequence
    X_unstacked = tf.unstack(X, axis=1)
    outputs = []
    current_state = initial_state
    
    # Loop computation
    for x_t in X_unstacked:
        # State update
        next_state = tf.matmul(current_state, A) + tf.matmul(x_t, B, transpose_b=True)
        
        # Output computation
        output_t = tf.matmul(next_state, C, transpose_b=True) + tf.matmul(x_t, D, transpose_b=True)
        
        if activation is not None:
            output_t = activation(output_t)
        
        outputs.append(output_t)
        current_state = next_state
    
    # Stack outputs
    outputs = tf.stack(outputs, axis=1)
    
    if return_states:
        return outputs, current_state
    else:
        return outputs


def ssm_layer(X, C2, state_dim=None, trainable=True, 
              kernel_initializer=tf.glorot_uniform_initializer(), 
              bias_initializer=tf.zeros_initializer(),
              activation=None,
              return_states=False):
    """
    State-Space Model (SSM) layer
    
    Parameters:
        X: Input tensor with shape [B, L, C1]
        C2: Output dimension
        state_dim: State dimension, default is C2
        trainable: Whether trainable (can be boolean or TensorFlow boolean tensor)
        kernel_initializer: Weight initializer
        bias_initializer: Bias initializer
        activation: Output activation function
        return_states: Whether to return state sequence
    
    Returns:
        Output tensor with shape [B, L, C2]
    """
    if state_dim is None:
        state_dim = C2
    
    # Get input dimensions
    input_shape = X.get_shape().as_list()
    batch_size = tf.shape(X)[0]  # Use batch_size to avoid conflict with parameter B
    seq_len = tf.shape(X)[1]
    C1 = input_shape[2] if input_shape[2] is not None else X.get_shape().as_list()[-1]
    
    # Convert string initializers to initializer objects
    if isinstance(kernel_initializer, str):
        if kernel_initializer == 'glorot_uniform' or kernel_initializer == 'xavier_uniform':
            kernel_initializer = tf.glorot_uniform_initializer()
        elif kernel_initializer == 'glorot_normal' or kernel_initializer == 'xavier_normal':
            kernel_initializer = tf.glorot_normal_initializer()
        elif kernel_initializer == 'he_uniform':
            kernel_initializer = tf.keras.initializers.he_uniform()
        elif kernel_initializer == 'he_normal':
            kernel_initializer = tf.keras.initializers.he_normal()
        elif kernel_initializer == 'zeros':
            kernel_initializer = tf.zeros_initializer()
        elif kernel_initializer == 'ones':
            kernel_initializer = tf.ones_initializer()
        elif kernel_initializer == 'random_normal':
            kernel_initializer = tf.random_normal_initializer()
        elif kernel_initializer == 'random_uniform':
            kernel_initializer = tf.random_uniform_initializer()
        else:
            kernel_initializer = tf.glorot_uniform_initializer()
    
    if isinstance(bias_initializer, str):
        if bias_initializer == 'zeros':
            bias_initializer = tf.zeros_initializer()
        elif bias_initializer == 'ones':
            bias_initializer = tf.ones_initializer()
        elif bias_initializer == 'random_normal':
            bias_initializer = tf.random_normal_initializer()
        elif bias_initializer == 'random_uniform':
            bias_initializer = tf.random_uniform_initializer()
        else:
            bias_initializer = tf.zeros_initializer()
    
    # Handle trainable parameter - if Tensor, create variables with Python boolean and use tf.cond when needed
    # tf.get_variable's trainable parameter must be Python boolean, not Tensor
    is_trainable_python = trainable if isinstance(trainable, bool) else True
    
    # SSM parameter initialization (A, B, C, D) - rename parameter matrices to avoid conflict
    # A: State transition matrix [state_dim, state_dim]
    A_matrix = tf.get_variable('A_matrix', shape=[state_dim, state_dim], 
                              initializer=kernel_initializer,
                              trainable=is_trainable_python)
    
    # B: Input matrix [state_dim, C1]
    B_matrix = tf.get_variable('B_matrix', shape=[state_dim, C1], 
                              initializer=kernel_initializer,
                              trainable=is_trainable_python)
    
    # C: Output matrix [C2, state_dim]
    C_matrix = tf.get_variable('C_matrix', shape=[C2, state_dim], 
                              initializer=kernel_initializer,
                              trainable=is_trainable_python)
    
    # D: Direct pass term [C2, C1]
    D_matrix = tf.get_variable('D_matrix', shape=[C2, C1], 
                              initializer=kernel_initializer,
                              trainable=is_trainable_python)
    
    # If trainable is Tensor, use tf.cond to control whether to use variables or their stop gradient version
    if not isinstance(trainable, bool):
        A_matrix = tf.cond(trainable, 
                          lambda: A_matrix, 
                          lambda: tf.stop_gradient(A_matrix))
        B_matrix = tf.cond(trainable, 
                          lambda: B_matrix, 
                          lambda: tf.stop_gradient(B_matrix))
        C_matrix = tf.cond(trainable, 
                          lambda: C_matrix, 
                          lambda: tf.stop_gradient(C_matrix))
        D_matrix = tf.cond(trainable, 
                          lambda: D_matrix, 
                          lambda: tf.stop_gradient(D_matrix))
    
    # Initial state - use correct shape handling
    initial_state = tf.zeros(tf.concat([tf.expand_dims(batch_size, 0), [state_dim]], axis=0), 
                            dtype=X.dtype)
    
    # Define one step computation of SSM
    def ssm_step(prev_state, inputs):
        # prev_state: [B, state_dim]
        # inputs: [B, C1]
        
        # State update: s_t = A * s_{t-1} + B * x_t
        next_state = tf.matmul(prev_state, A_matrix) + tf.matmul(inputs, B_matrix, transpose_b=True)
        
        # Output computation: y_t = C * s_t + D * x_t
        output = tf.matmul(next_state, C_matrix, transpose_b=True) + tf.matmul(inputs, D_matrix, transpose_b=True)
        
        return next_state, output
    
    # Handle static sequence length
    static_seq_len = input_shape[1]
    
    if static_seq_len is not None and static_seq_len > 0:
        # Static sequence length, use loop
        X_unstacked = tf.unstack(X, axis=1)
        outputs = []
        current_state = initial_state
        
        for t in range(static_seq_len):
            current_state, output_t = ssm_step(current_state, X_unstacked[t])
            if activation is not None:
                output_t = activation(output_t)
            outputs.append(output_t)
        
        outputs = tf.stack(outputs, axis=1)
    else:
        # Dynamic sequence length, use tf.while_loop
        def loop_body(t, current_state, outputs_ta):
            # Get input of current time step
            x_t = tf.gather(X, t, axis=1)
            
            # SSM computation
            next_state, output_t = ssm_step(current_state, x_t)
            
            if activation is not None:
                output_t = activation(output_t)
            
            # Store output
            outputs_ta = outputs_ta.write(t, output_t)
            
            return t + 1, next_state, outputs_ta
        
        # Create TensorArray to store outputs
        outputs_ta = tf.TensorArray(dtype=X.dtype, size=seq_len, dynamic_size=False)
        
        # Execute loop
        _, final_state, outputs_ta = tf.while_loop(
            cond=lambda t, *_: t < seq_len,
            body=loop_body,
            loop_vars=(tf.constant(0, dtype=tf.int32), initial_state, outputs_ta),
            parallel_iterations=1
        )
        
        # Convert TensorArray to tensor
        outputs = outputs_ta.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])  # [L, B, C2] -> [B, L, C2]
    
    if return_states:
        if static_seq_len is not None and static_seq_len > 0:
            return outputs, current_state
        else:
            return outputs, final_state
    else:
        return outputs


def cross_attention_sum(X1, X2, C1, key_masks1=None, key_masks2=None, dropout=0.0, 
                       training=True, scope="cross_attention_sum", 
                       initializer=tf.glorot_uniform_initializer(),
                       use_residual=True, activation=None):
    """
    Sum after mutual attention between X1 and X2
    
    Parameters:
        X1: Input tensor 1 with shape [B, L, C]
        X2: Input tensor 2 with shape [B, L, C]
        C1: Output dimension
        key_masks1: Mask tensor for X1 with shape [B, L] or [B, 1, L]
        key_masks2: Mask tensor for X2 with shape [B, L] or [B, 1, L]
        dropout: Dropout rate
        training: Whether in training mode
        scope: Variable scope
        initializer: Weight initializer
        use_residual: Whether to use residual connection
        activation: Output activation function
    
    Returns:
        Output tensor with shape [B, L, C1]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Get input dimensions
        B = tf.shape(X1)[0]
        L = tf.shape(X1)[1]
        C = X1.get_shape().as_list()[2]
        
        # Ensure X1 and X2 have matching shapes
        assert X1.get_shape().as_list()[-1] == X2.get_shape().as_list()[-1], \
            "X1 and X2 must have the same last dimension"
        
        # Create all-ones masks if not provided
        if key_masks1 is None:
            key_masks1 = tf.ones([B, L], dtype=tf.float32)
        if key_masks2 is None:
            key_masks2 = tf.ones([B, L], dtype=tf.float32)
        
        # Adjust mask shapes to [B, 1, L]
        if len(key_masks1.get_shape()) == 2:
            key_masks1 = tf.expand_dims(key_masks1, axis=1)
        if len(key_masks2.get_shape()) == 2:
            key_masks2 = tf.expand_dims(key_masks2, axis=1)
        
        # Linear transformation to generate Q, K, V (shared projection layers)
        d_model = C1  # Projection dimension
        
        # X1 -> X2 attention (X1 attends to X2)
        Q1 = tf.layers.dense(X1, d_model, kernel_initializer=initializer, name="query1")
        K2 = tf.layers.dense(X2, d_model, kernel_initializer=initializer, name="key2")
        V2 = tf.layers.dense(X2, C1, kernel_initializer=initializer, name="value2")
        
        # Apply dropout
        Q1 = tf.layers.dropout(Q1, rate=dropout, training=training)
        K2 = tf.layers.dropout(K2, rate=dropout, training=training)
        V2 = tf.layers.dropout(V2, rate=dropout, training=training)
        
        # Attention output of X1 attending to X2
        attn1_to_2 = AttentionCapsuleScaleDot(Q1, K2, V2, key_masks2, 
                                             dropout=dropout, training=training,
                                             scope="attention_x1_to_x2")
        
        # X2 -> X1 attention (X2 attends to X1)
        Q2 = tf.layers.dense(X2, d_model, kernel_initializer=initializer, name="query2")
        K1 = tf.layers.dense(X1, d_model, kernel_initializer=initializer, name="key1")
        V1 = tf.layers.dense(X1, C1, kernel_initializer=initializer, name="value1")
        
        # Apply dropout
        Q2 = tf.layers.dropout(Q2, rate=dropout, training=training)
        K1 = tf.layers.dropout(K1, rate=dropout, training=training)
        V1 = tf.layers.dropout(V1, rate=dropout, training=training)
        
        # Attention output of X2 attending to X1
        attn2_to_1 = AttentionCapsuleScaleDot(Q2, K1, V1, key_masks1, 
                                             dropout=dropout, training=training,
                                             scope="attention_x2_to_x1")
        
        # Sum the two attention outputs
        attn_sum = attn1_to_2 + attn2_to_1
        
        # Output projection and fusion
        output = tf.layers.dense(attn_sum, C1, kernel_initializer=initializer, name="output_fusion")
        
        # Apply dropout
        output = tf.layers.dropout(output, rate=dropout, training=training)
        
        # Residual connection (using X1 as residual)
        if use_residual:
            if C != C1:
                X1_proj = tf.layers.dense(X1, C1, kernel_initializer=initializer, name="residual_projection")
                output = output + X1_proj
            else:
                output = output + X1
        
        # Layer normalization
        output = tf.contrib.layers.layer_norm(output, begin_norm_axis=-1)
        
        # Activation function
        if activation is not None:
            output = activation(output)
        
        return output


# Enhanced version: includes self-attention + cross-attention
def cross_self_attention_sum(X1, X2, C1, key_masks1=None, key_masks2=None, dropout=0.0, 
                           training=True, scope="cross_self_attention_sum",
                           initializer=tf.glorot_uniform_initializer(),
                           use_residual=True, activation=None):
    """
    Sum after mutual attention + self-attention between X1 and X2
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Get input dimensions
        B = tf.shape(X1)[0]
        L = tf.shape(X1)[1]
        C = X1.get_shape().as_list()[2]
        
        # Self-attention
        X1_self = tf.layers.dense(X1, d_model, kernel_initializer=initializer, name="self_query1")
        X1_self_attn = AttentionCapsuleScaleDot(X1_self, X1_self, 
                                               tf.layers.dense(X1, C1, name="self_value1"), 
                                               key_masks1, dropout=dropout, training=training,
                                               scope="self_attention_x1")
        
        X2_self = tf.layers.dense(X2, d_model, kernel_initializer=initializer, name="self_query2")
        X2_self_attn = AttentionCapsuleScaleDot(X2_self, X2_self, 
                                               tf.layers.dense(X2, C1, name="self_value2"), 
                                               key_masks2, dropout=dropout, training=training,
                                               scope="self_attention_x2")
        
        # Cross-attention
        cross_attn = cross_attention_sum(X1, X2, C1, key_masks1, key_masks2, 
                                        dropout=dropout, training=training,
                                        scope="cross_attention_only",
                                        use_residual=False, activation=None)
        
        # Weighted sum of all attention outputs
        alpha = tf.get_variable("alpha", shape=[1], initializer=tf.constant_initializer(0.33),
                               trainable=True)
        beta = tf.get_variable("beta", shape=[1], initializer=tf.constant_initializer(0.33),
                              trainable=True)
        gamma = tf.get_variable("gamma", shape=[1], initializer=tf.constant_initializer(0.33),
                               trainable=True)
        
        output = alpha * X1_self_attn + beta * X2_self_attn + gamma * cross_attn
        
        # Post processing
        if use_residual:
            if C != C1:
                X_proj = tf.layers.dense((X1 + X2) / 2, C1, kernel_initializer=initializer, name="residual_projection")
                output = output + X_proj
            else:
                output = output + (X1 + X2) / 2
        
        output = tf.contrib.layers.layer_norm(output, begin_norm_axis=-1)
        
        if activation is not None:
            output = activation(output)
        
        return output




def attentionalMambaBlock(X,channels,strides=2,dropout=0.5,is_train=True,scope="value"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        padding = "SAME"
        with tf.variable_scope("headA"):
            head_1 = SquashCNNs(X,channels,9,strides,padding,is_train)
            mask_1 =  tf.math.equal(head_1,0)
            S_X_1 =self_attention_layer(head_1, channels, key_masks=mask_1, dropout=dropout, training=is_train, scope="self_attention", initializer=tf.glorot_uniform_initializer(),use_residual=True, activation=None)
            S_X_1 = ln(S_X_1)
        with tf.variable_scope("headB"):
            head_2 = SquashCNNs(X,channels,7,strides,padding,is_train)
            S_X_2 = ssm_layer(head_2,channels,state_dim=channels, trainable=is_train, kernel_initializer='glorot_uniform', activation=squash)
            S_X_2 =  ln(S_X_2)
        with tf.variable_scope("mutual_cross_attention"):
            c1_mask =  tf.math.equal(S_X_1,0)
            c2_mask =  tf.math.equal(S_X_2,0)
            outputs  = cross_attention_sum(S_X_1,S_X_2,channels,key_masks1=c1_mask, key_masks2=c2_mask, dropout=dropout ,training=is_train, scope="cross_attention_sum", initializer=tf.glorot_uniform_initializer(),use_residual=True, activation=None)
            return outputs


blovknum = 3
channelslist = [128,256,256]
stridelist=[2,2,2]


def KCapMamba(X,num_classes,dropout=0.5,is_train=True):
    with tf.variable_scope("body"):
        input_variable = X
        for i in range(len(channelslist)):
            scope = "block_"+str(i+1)
            input_variable = attentionalMambaBlock(input_variable,channelslist[i],stridelist[i],dropout,is_train,scope)
    with tf.variable_scope("routing"):
            block_listnum = channelslist[-1]
            routing_head = 8
            trans_capsule_shape = get_shape(input_variable)
            trans_fc_function = tf.reshape(input_variable,shape=(-1,int(block_listnum//routing_head),1,routing_head*trans_capsule_shape[1],1))
            trans_blu = tf.zeros([trans_capsule_shape[0],int(block_listnum//routing_head),num_classes,1,1])
            trans_caps = routing(trans_fc_function,trans_blu,num_classes,num_dims=routing_head*2)
    with tf.variable_scope("ConcatenatedLayer"):
        caps = tf.concat([trans_caps],axis=3)
        caps = tf.layers.dropout(caps,rate=dropout,training=is_train)
    with tf.variable_scope("Prediction"):
        caps = tf.squeeze(caps,axis=1)
        v_length = tf.sqrt(reduce_sum(tf.square(caps),axis=2,keepdims=True)+eposilion)
        v_length = tf.reshape(v_length,shape=[-1,num_classes  ])
    return v_length
        












capsule_coefficient = 0.9
lamdaset = 0.5
def margin_loss(v_length,Y,name="Teacher",weight_decay_rate=0.0005):
    with tf.variable_scope(name+"margin_loss"):
        max_l = tf.square(tf.maximum(0.0,capsule_coefficient-v_length))
        max_r = tf.square(tf.maximum(0.0,(v_length-(1-capsule_coefficient))))
        margin = tf.reduce_mean( tf.reduce_sum(Y * max_l  + lamdaset * (1-Y) * max_r,axis=1))
        margin = decay_weights(margin,weight_decay_rate)
        return margin
