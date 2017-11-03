import tensorflow as tf
import numpy as np

def l2_reg(lam):
    l = 0.0
    train_vars = [v for v in tf.trainable_variables() if 'W' in v.name]
    N = len(train_vars)
    for v in train_vars:
        l += (lam/N)*tf.reduce_mean(tf.square(v))
    return l

def conv2D(x, dims=[3,3], nfilters=32, strides=[1,1],
           init=1e-3, padding='SAME', activation=tf.identity, scope='conv2d', reuse=False):
    """
    args:
        x, (tf tensor), tensor with shape (batch,width,height,channels)
        dims, (list), size of convolution filters
        filters, (int), number of filters used
        strides, (list), number of steps convolutions slide
        std, (float/string), std of weight initialization, 'xavier' for xavier
            initialization
        padding, (string), 'SAME' or 'VALID' determines if input should be padded
            to keep output dimensions the same or not
        activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
        scope, (string), scope under which to store variables
        reuse, (boolean), whether we want to reuse variables that have already
            been created (i.e. reuse an earilier layer)
    returns:
        a, (tf tensor), the output of the convolution layer, has size
            (batch, new_width , new_height , filters)
    """
    with tf.variable_scope(scope,reuse=reuse):

        s = x.get_shape().as_list()

        shape = dims +[s[3],nfilters]

        if init=='xavier':
            init = np.sqrt(2.0/(s[1]*s[2]*s[3]))

        W = tf.Variable(tf.random_normal(shape=shape,stddev=init),
            name='W')
        b = tf.Variable(tf.ones([nfilters])*init, name='b')

        o = tf.nn.convolution(x, W, padding, strides=strides)

        o = o+b

        a = activation(o)

        return a

def fullyConnected(x,output_units=100,activation=tf.identity,std=1e-3,
                  scope='fc',reuse=False):
    """
    args:
        x, (tf tensor), tensor with shape (batch,width,height,channels)
        std, (float/string), std of weight initialization, 'xavier' for xavier
            initialization
        output_units,(int), number of output units for the layer
        activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
        scope, (string), scope under which to store variables
        reuse, (boolean), whether we want to reuse variables that have already
            been created (i.e. reuse an earilier layer)
    returns:
        a, (tf tensor), the output of the fullyConnected layer, has size
            (batch, output_units)
    """
    with tf.variable_scope(scope,reuse=reuse):

        s = x.get_shape().as_list()
        print s
        shape = [s[1],output_units]

        if std=='xavier':
            std = np.sqrt(2.0/shape[0])

        W = tf.get_variable('W',shape=shape,initializer=tf.random_normal_initializer(0.0,std))
        b = tf.get_variable("b",shape=shape[1],initializer=tf.random_normal_initializer(0.0,std))

        h = tf.matmul(x,W)+b
        a = activation(h)
        return a

def repeat(x,axis,repeat):
    s = x.get_shape().as_list()
    splits = tf.split(value=x,num_or_size_splits=s[axis],axis=axis)
    rep = [s for s in splits for _ in range(repeat)]
    return tf.concat(rep,axis)

def resize_tensor(x,scales=[1,2,2,1]):
    out = x
    for i in range(1,len(scales)):
        out = repeat(out,i,scales[i])
    return out

def upsample2D(x, scope='upsample'):
    with tf.variable_scope(scope):
        #o = resize_tensor(x)
        s = x.get_shape().as_list()
        s = [s[1]*2,s[2]*2]
        #Defaults to bilinear
        o = tf.image.resize_images(x,s)
        return o

def I2INetBlock(x,nfilters=32,init='xavier',activation=tf.nn.relu, num=2,scope='i2i'):
    with tf.variable_scope(scope):
        o1 = conv2D(x,nfilters=nfilters,init=init,activation=activation)
        o2 = conv2D(o1,nfilters=nfilters,init=init,activation=activation)
        if num==3:
            o2 = conv2D(o1,nfilters=nfilters,init=init,activation=activation)
        o3 = tf.nn.pool(o2,[2,2],strides=[2,2],pooling_type='AVG',padding='SAME',name='pool')
    return o3,o2

def I2INetUpsampleBlock(x,y,n1=64,n2=32,init='xavier',activation=tf.nn.relu,scope='ublock'):
    with tf.variable_scope(scope):
        y = upsample2D(y)
        x = tf.concat([x,y],axis=3)
        o = conv2D(x,dims=[1,1],nfilters=n1,init=init,activation=activation)
        o1 = conv2D(o,nfilters=n2,init=init,activation=activation)
        o2 = conv2D(o1,nfilters=n2,init=init,activation=activation)

        return o2

def I2INet(x, nfilters=32, init='xavier', activation=tf.nn.relu):

    d1,a1 = I2INetBlock(x,nfilters=nfilters,init=init,activation=activation,scope='block1')
    d2,a2 = I2INetBlock(d1,nfilters=4*nfilters,init=init,activation=activation, scope='block2')
    d3,a3 = I2INetBlock(d2,nfilters=8*nfilters,init=init,activation=activation, scope='block3')
    d4,a4 = I2INetBlock(d3,nfilters=16*nfilters,init=init,activation=activation, scope='block4')

    o4 = I2INetUpsampleBlock(a3,a4,n1=16*nfilters,n2=8*nfilters,init=init,activation=activation, scope='ublock1')
    o3 = I2INetUpsampleBlock(a2,o4,n1=8*nfilters,n2=4*nfilters,init=init,activation=activation, scope='ublock2')
    o2 = I2INetUpsampleBlock(a1,o3,n1=4*nfilters,n2=nfilters,init=init,activation=activation, scope='ublock3')

    out = conv2D(o2,nfilters=1,activation=tf.identity)
    out_class = tf.sigmoid(out)
    return out_class,out,o3,o4

def class_balanced_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    Args:
        pred: of shape (b, ...). the predictions in [0,1].
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_cross_entropy'):
        z = batch_flatten(pred)
        y = tf.cast(batch_flatten(label), tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        eps = 1e-12
        loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
        loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
    cost = tf.subtract(loss_pos, loss_neg, name=name)
    return cost


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)
