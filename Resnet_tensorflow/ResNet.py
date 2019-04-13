
import tensorflow as tf

CONV_WEIGHT_STDDEV=0.04
CONV_WEIGHT_DECAY=0.0004
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01

def _variable_weight(_name="weight",_shape=None,_stddev=0.02,_wd=None):
        _intinitializer=tf.truncated_normal_initializer(stddev=_stddev)
        weight=tf.get_variable(name=_name,shape=_shape,initializer=_intinitializer)

        if(_wd!=None):
            #正则化系数，如果该参数不为None,则对参数 W 进行正则，并且将正则化添加到损失函数当中；
            weight_decay =tf.multiply(tf.nn.l2_loss(weight), _wd, name='weight_loss')
            tf.add_to_collection('loss', weight_decay)
        return weight

def conv(name,x,kernel_w,kernel_h,stride_w,stride_h,out_dim,pad='SAME'):
    with tf.variable_scope(name) as scope:
        in_dim=x.get_shape()[-1]

        w_shape=[kernel_w,kernel_h,in_dim,out_dim]
        weight=_variable_weight(_shape=w_shape,_stddev=CONV_WEIGHT_DECAY,_wd=CONV_WEIGHT_DECAY)
        # bias=tf.get_variable(name='bias',shape=[out_dim],initializer=tf.constant_initializer(0.0001))
        out=tf.nn.conv2d(x,weight,[1,stride_w,stride_h,1],padding=pad,name=scope.name)

        return out

def bn(_name,x,_trainable):
    out=tf.layers.batch_normalization(x,
                                axis=3, #if data_format == 'channels_first' else 3
                                momentum=0.997,
                                epsilon=1e-5,
                                center=True,
                                scale=True,
                                trainable=_trainable,
                                fused=True,
                                name=_name)
    return out

def fc(name,x,out_dim):

    with tf.variable_scope(name) as scope:

        in_dim=x.get_shape()[-1]

        w_shape=[in_dim,out_dim]
        weight=_variable_weight(_shape=w_shape,_stddev=FC_WEIGHT_STDDEV,_wd=FC_WEIGHT_DECAY)
        bias=tf.get_variable('bias',shape=[out_dim],initializer=tf.zeros_initializer)
        out=tf.nn.xw_plus_b(x,weight,bias)
        return out

def max_pool(x,ksize,stride,pad='SAME'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=pad)

def shortcut(x,out_dim,ksize=1,stride=1,_trainable=True):
    with tf.variable_scope('shortcut') as scope:
        out=conv(scope.name+'_conv',x,ksize,ksize,stride,stride,out_dim)
        out=bn('shortcut_bn',out,_trainable)
        return out

def conv_bn_relu(x,_ksize,_stride,out_dim,_trainable,pad='SAME'):
    with tf.variable_scope('conv_bn_relu') as scope:
        out=conv(scope.name+'conv',x,_ksize,_ksize,_stride,_stride,out_dim,pad=pad)
        out=bn(scope.name+'bn',out,_trainable=_trainable)
        out=tf.nn.relu(out)
        return out

def block(x,c):
    in_dim=x.get_shape()[-1]
    out_dim=c['out_dim']
    ksize=c['ksize']
    stride=c['stride']
    out_shortcut=x
    with tf.variable_scope('a'):
        out=conv_bn_relu(x,1,stride,out_dim,c['trainable'])
    
    with tf.variable_scope('b'):
        out=conv_bn_relu(out,ksize,1,out_dim,c['trainable'])
    
    with tf.variable_scope('c'):
        out=conv_bn_relu(out,1,1,out_dim*4,c['trainable'])
    
    if(in_dim!=(out_dim*4)or(stride!=1)):
        out_shortcut=shortcut(x,out_dim*4,1,stride,c['trainable'])
    
    return(out+out_shortcut)

def stack(x,c):
    for n in range(c['num_block']):
        s=c['stride'] if n==0 else 1
        c['stride']=s
        with tf.variable_scope('block%d'%(n)):
            x=block(x,c)
    return x

def inference_resnet_50(x,trainable,num_class):
    
    c={}
    c['trainable']=trainable

    x=conv('conv1',x,7,7,2,2,64)
    x=max_pool(x,3,2)

    with tf.variable_scope('conv2_x'):
        c['ksize']=3
        c['stride']=1
        c['num_block']=3
        c['out_dim']=64
        x=stack(x,c)
    
    with tf.variable_scope('conv3_x'):
        c['ksize']=3
        c['stride']=2
        c['num_block']=4
        c['out_dim']=128
        x=stack(x,c)
    
    with tf.variable_scope('conv4_x'):
        c['ksize']=3
        c['stride']=2
        c['num_block']=6
        c['out_dim']=256
        x=stack(x,c)
    
    with tf.variable_scope('conv5_x'):
        c['ksize']=3
        c['stride']=2
        c['num_block']=3
        c['out_dim']=512
        x=stack(x,c)
    
    x=tf.nn.avg_pool(x,[1,7,7,1],[1,1,1,1],padding='VALID',name='avg_pool')
    dim=x.get_shape()[-1]
    x=tf.reshape(x,[-1,dim])
    x=fc('fc',x,num_class)

    return x











