# -*- coding: UTF-8 -*-

import gluonbook as gb
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import os, time, shutil
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

def try_gpu4():
    try:
        ctx=mx.gpu()
        _=nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx=mx.cpu()
    # ctx=mx.cpu()
    return ctx

def evaluate_accuracy(data_iter,net,ctx):
    test_time_begin=time.time()
    acc=nd.array([0],ctx=ctx)
    for X,y in data_iter:
        X,y=X.as_in_context(ctx),y.as_in_context(ctx)
        acc+=gb.accuracy(net(X),y)
    test_time_end=time.time()
    test_time=test_time_end-test_time_begin
    test_acc=acc.asscalar()/len(data_iter)
    return test_acc,test_time

def train_module(net_name,net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
    print('training on',ctx)

    start=time.time()
    test_acc_old=0.0
    loss=gloss.SoftmaxCrossEntropyLoss()
    early_stop=15
    stop_index=0
    for epoch in range(num_epochs):
        if(stop_index>early_stop):
            break
        train_l_sum,train_acc_sum=0,0
        for X,y in train_iter:
            X,y=X.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                y_hat=net(X)
                l=loss(y_hat,y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum+=l.mean().asscalar()
            train_acc_sum+=gb.accuracy(y_hat,y)
        test_acc,test_time=evaluate_accuracy(test_iter,net,ctx)
        if test_acc>test_acc_old:
            # net.save_parameters('{:s}_best.params'.format(net_name))
            test_acc_old=test_acc
            print('best test acc ={:.4f}'.format(test_acc))
            stop_index = 0
        else:
            stop_index += 1


        loss_acc=train_l_sum/len(train_iter)
        train_acc=train_acc_sum/len(train_iter)
        print('epoch={:d},loss={:.4f},train_acc={:.4f},test_acc={:.4f},test_time={:.2f}'.format(epoch+1,loss_acc,train_acc,test_acc,test_time))
        if((epoch+1)%10==0):
            train_time=time.time()-start
            print('epoch %d, train_time %.2f'%(epoch+1,train_time))


if __name__=='__main__':


    jitter_param = 0.4
    lighting_param = 0.1
    batch_size =10
    num_workers=1
    lr=0.005
    momentum=0.9
    num_epochs =300
    classes =80

    # 数据处理部分###############################################
    path = 'D:/HFUT-VL-Dataset-3/'
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')

    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(299),   #inceptionv3 input size;
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    ######################################################


    ctx=try_gpu4()
    model_name = 'densenet121'
    net = get_model(model_name, pretrained=True)
    net.output = nn.Dense(classes)
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr,'momentum': momentum})
    train_module(model_name,net, train_data, test_data, batch_size, trainer, ctx, num_epochs)


