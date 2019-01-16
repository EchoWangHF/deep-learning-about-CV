import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

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

