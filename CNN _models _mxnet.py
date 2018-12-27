import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

def LeNet():
    net=nn.Sequential()
    net.add(
        nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10)    
    )
    return net

def AlexNet():
    net=nn.Sequential()
    net.add(
        nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),

        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),

        nn.Dense(4096,activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10)
    )
    return net

def VGGNet(conv_arch=None,outchannels=10):
    if conv_arch==None:
        # 默认为VGG16
        conv_arch=((1,32), (1,64))
    
    net=nn.Sequential()
    for num_convs,num_channels in conv_arch:
        # VGG block部分
        blk=nn.Sequential()
        for _ in range(num_convs):
            blk.add(nn.Conv2D(num_channels,kernel_size=3,padding=1,activation='relu'))
            blk.add(nn.MaxPool2D(pool_size=2,strides=2))
        net.add(blk)
    
    net.add(
        nn.Dense(4096,activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(outchannels)
    )
    return net

# NIN Net部分 :Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
def NiN_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(
        nn.Conv2D(num_channels, kernel_size,strides, padding, activation='relu'),
        nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
        nn.Conv2D(num_channels, kernel_size=1, activation='relu') #去除一个1x1的卷积层，效果差到感人。
    )
    return blk

def NiNNet(outchannels):
    net=nn.Sequential()
    net.add(
        NiN_block(96,kernel_size=11,strides=4,padding=0),
        nn.MaxPool2D(pool_size=3,strides=2),
        NiN_block(256,kernel_size=5,strides=1,padding=1),
        nn.MaxPool2D(pool_size=3,strides=2),
        NiN_block(384,kernel_size=3,strides=1,padding=1),
        nn.MaxPool2D(pool_size=3,strides=2),
        NiN_block(outchannels,kernel_size=3,strides=1,padding=1),
        nn.GlobalAvgPool2D(),
        nn.Flatten()
    )
    return net

#GoogleNet 部分:
class Inception(nn.Block):
    def __init__(self,c1,c2,c3,c4,**kwargs):
        # c1,c2,c3,c4分别表示inception模板当中四条线路输出的通道数
        super(Inception,self).__init__(**kwargs)

        # 线路1，1x1 的卷积层
        self.p1_1=nn.Conv2D(c1,kernel_size=1,activation='relu')
        # 线路2,1×1卷积之后，接3x3的卷积
        self.p2_1=nn.Conv2D(c2[0],kernel_size=1,activation='relu')
        self.p2_2=nn.Conv2D(c2[1],kernel_size=3,padding=1,activation='relu')
        # 线路3,1×1卷积之后，接5x5的卷积
        self.p3_1=nn.Conv2D(c3[0],kernel_size=1,activation='relu')
        self.p3_2=nn.Conv2D(c3[1],kernel_size=5,padding=2,activation='relu')
        # 线路4,3x3最大池化层之后，接1×1卷积
        self.p4_1=nn.MaxPool2D(pool_size=3,strides=1,padding=1)
        self.p4_2=nn.Conv2D(c4,kernel_size=1,activation='relu')
    
    def forward(self,x):
        p1=self.p1_1(x)
        p2=self.p2_2(self.p2_1(x))
        p3=self.p3_2(self.p3_1(x))
        p4=self.p4_2(self.p4_1(x))

        return nd.concat(p1,p2,p3,p4,dim=1) #在通道维度上面进行链接

def GoogLeNet(outchannels):
    b1=nn.Sequential()
    b1.add(
        nn.Conv2D(64,kernel_size=7,strides=2,padding=3,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1)
    )

    b2=nn.Sequential()
    b2.add(
        nn.Conv2D(64,kernel_size=1),
        nn.Conv2D(192,kernel_size=3,padding=1),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1)
    )

    b3=nn.Sequential()
    b3.add(
        Inception(64,(96,128),(16,32),32),
        Inception(128,(128,192),(32,96),64),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1)
    )

    b4 = nn.Sequential()
    b4.add(
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    b5=nn.Sequential()
    b5.add(
        Inception(256,(160,320),(32,128),128),
        Inception(384,(192,384),(48,128),128),
        nn.GlobalAvgPool2D()
    )

    net=nn.Sequential()
    net.add(b1,b2,b3,b4,b5,nn.Dense(outchannels))
    return net

#Batch Normalize层的相关操作
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not autograd.is_training():
        # 如果是在预测模式下,直接使用传入的移动平均所得的均值和方差。
        x_hat=(X-moving_mean)/nd.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            # 使用全连接层的情况,计算特征维上的均值和方差。
            mean=X.mean(axis=0)
            var=((X-mean)**2).mean(axis=0)
        else:
            # 使用二维卷积层的情况,计算通道维上(axis=1)的均值和方差。这里我们需要
            # 保持 X 的形状以便后面可以做广播运算。
            mean=X.mean(axis=(0,2,3),keepdims=True)
            var=((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        x_hat=(X-mean)/nd.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var
    Y=gamma*x_hat+beta
    return Y,moving_mean,moving_var

class BatchNorm(nn.Block):
    def __init__(self,num_features,num_dims,**kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        
        self.gamma=self.params.get('gamma',shape=shape,init=init.One())
        self.beta=self.params.get('beta',shape=shape,init=init.Zero())
        self.moving_mean=nd.zeros(shape)
        self.moving_var=nd.zeros(shape)
    
    def forward(self,X):
        if self.moving_mean.context!=X.context:
            self.moving_mean=self.moving_mean.copyto(X.context)
            self.moving_var=self.moving_var.copyto(X.context)
        Y,self.moving_mean,self.moving_var=batch_norm(X,self.gamma.data(),self.beta.data(),self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y

def AlexNet_BN():
    net=nn.Sequential()
    net.add(
        nn.Conv2D(96,kernel_size=11,strides=4),
        BatchNorm(96,num_dims=4),
        nn.Activation('relu'),

        nn.MaxPool2D(pool_size=3,strides=2),

        nn.Conv2D(256,kernel_size=5,padding=2),
        BatchNorm(256,num_dims=4),
        nn.Activation('relu'),

        nn.MaxPool2D(pool_size=3,strides=2),

        nn.Conv2D(384,kernel_size=3,padding=1),
        BatchNorm(384,num_dims=4),
        nn.Activation('relu'),

        nn.Conv2D(384,kernel_size=3,padding=1),
        BatchNorm(384,num_dims=4),
        nn.Activation('relu'),

        nn.Conv2D(256,kernel_size=3,padding=1),
        BatchNorm(256,num_dims=4),
        nn.Activation('relu'),

        nn.MaxPool2D(pool_size=3,strides=2),

        nn.Dense(4096),
        BatchNorm(4096,num_dims=2),
        nn.Activation('relu'),
        nn.Dropout(0.5),

        nn.Dense(4096),
        BatchNorm(4096,num_dims=2),
        nn.Activation('relu'),
        nn.Dropout(0.5),

        nn.Dense(10)
    )
    return net

# ResNet相关部分
class Residual(nn.Block):
    def __init__(self,num_channels,use_1x1conv=False,strides=1,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1=nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=strides)
        self.conv2=nn.Conv2D(num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm()
        self.bn2=nn.BatchNorm()

    def forward(self,X):
        Y=nd.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        return nd.relu(Y+X)

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
    
def ResNet(outchannels):
    net=nn.Sequential()
    net.add(
        nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2),
        nn.GlobalAvgPool2D(), 
        nn.Dense(outchannels)
    )
    return net

# DenseNet 相关部分
def conv_block(num_channels):
    blk=nn.Sequential()
    blk.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(num_channels,kernel_size=3,padding=1)
    )
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))
    
    def forward(self,X):
        for blk in self.net:
            Y=blk(X)
            X=nd.concat(X,Y,dim=1)
        return X

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(
        nn.BatchNorm(), 
        nn.Activation('relu'),
        nn.Conv2D(num_channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return blk

def DenseNet(outchannels):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), 
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    num_channels, growth_rate = 64, 32 # num_channels:当前的通道数。
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密的输出通道数。
        num_channels += num_convs * growth_rate
        # 在稠密块之间加入通道数减半的过渡层。
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add(transition_block(num_channels // 2))

    net.add(
        nn.BatchNorm(), 
        nn.Activation('relu'), 
        nn.GlobalAvgPool2D(),
        nn.Dense(outchannels)
    )
    return net


def try_gpu4():
    try:
        ctx=mx.gpu()
        _=nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx=mx.cpu()
    return ctx

def evaluate_accuracy(data_iter,net,ctx):
    acc=nd.array([0],ctx=ctx)
    for X,y in data_iter:
        X,y=X.as_in_context(ctx),y.as_in_context(ctx)
        acc+=gb.accuracy(net(X),y)
    return acc.asscalar()/len(data_iter)

def train_module(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
    print('training on',ctx)
    loss=gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,start=0,0,time.time()
        for X,y in train_iter:
            X,y=X.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                y_hat=net(X)
                l=loss(y_hat,y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum+=l.mean().asscalar()
            train_acc_sum+=gb.accuracy(y_hat,y)
        test_acc=evaluate_accuracy(test_iter,net,ctx)
        print('epoch %d,loss%.4f,train acc %.3f,test acc %.3f,time %.1f sec'%(epoch+1,train_l_sum/len(train_iter),
              train_acc_sum/len(train_iter),test_acc,time.time()-start))

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,resize=96)
lr, num_epochs = 0.05, 5
net=DenseNet(10)
ctx=try_gpu4()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_module(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)



