import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

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
net=NiNNet(10)
ctx=try_gpu4()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_module(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

