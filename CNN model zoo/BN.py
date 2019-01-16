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
