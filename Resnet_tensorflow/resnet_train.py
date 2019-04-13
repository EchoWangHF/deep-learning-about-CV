import tensorflow as tf
import numpy as np
import TFRecord
import ResNet
import Param

def train():
    with tf.name_scope('data'):
        train_x,train_y=TFRecord.Read_Train_TFRecord()
        test_x,test_y=TFRecord.Read_Val_TFRecord()
    
    # net=Net.CNN_Net(Param.Class_Num,Param.Width,Param.Height,Param.Channel)
    # output,x,y=net.GoogleNet()

    with tf.name_scope('input_data'):
        x = tf.placeholder(tf.float32, shape=[None, Param.Width,Param.Height,Param.Channel], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, Param.Class_Num], name='Output')
    
    output=ResNet.inference_resnet_50(x,True,Param.Class_Num)

    #动态调整learning rate
    global_step=tf.Variable(0,False)
    starter_learning_rate=0.001
    Learning_Rate=tf.train.exponential_decay(starter_learning_rate, global_step,
                                             50000, 0.9, staircase=True)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)+tf.add_n(tf.get_collection('loss')))
    # loss=-tf.reduce_mean(tf.matmul(y,tf.log(tf.clip_by_value(output,1e-8,1.0))))
    optimizer = tf.train.AdamOptimizer(learning_rate=Learning_Rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)

    y_test=tf.nn.softmax(output)
    correct_prediction=tf.equal(tf.argmax(y_test,1),tf.argmax(y,1))
    accurcy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #save model
    check_path = "D:/Model/ImageNet/ResNet_model.ckpt"
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, check_path)
        # saver.save(sess, check_path)
        # saver.restore(sess, check_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        i=1
        while(True):
            train_images,train_labels=sess.run([train_x,train_y])

            # t=sess.run(output,feed_dict={x:train_images,y:train_labels})
            # print(t.shape)
        
            sess.run([optimizer],feed_dict={x:train_images,y:train_labels})

            if((i%50)==0):
                train_images,train_labels=sess.run([train_x,train_y])
                train_loss,train_acc=sess.run([loss,accurcy],feed_dict={x:train_images,y:train_labels})
                print(' i=%d, train loss=%5f, train accurcy=%.5f'%(i,train_loss,train_acc))

            if ((i%500)==0):
                test_images,test_labels=sess.run([test_x,test_y])
                test_loss,test_acc=sess.run([loss,accurcy],feed_dict={x:test_images,y:test_labels})
                print('************ i=%d, Test loss=%5f, test accurcy=%.5f'%(i,test_loss,test_acc))

            if ((i%5000)==0):
                saver.save(sess, check_path)
            i+=1
        coord.request_stop()
        coord.join(threads)



if(__name__=='__main__'):
    train()

