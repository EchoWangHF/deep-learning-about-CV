import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

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
net=GoogLeNet(10)
ctx=try_gpu4()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_module(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

