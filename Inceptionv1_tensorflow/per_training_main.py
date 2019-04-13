import tensorflow as tf
import numpy as np
import TF_Records
import Net
import Param

per_train_check_path = "D:/Model/ImageNet/GoogleNet_model.ckpt"

def train():
    with tf.name_scope('data'):
        train_x,train_y=TF_Records.TFRecord_Read('train')
        test_x,test_y=TF_Records.TFRecord_Read('test')
    
    net=Net.CNN_Net(Param.Class_Num,Param.Width,Param.Height,Param.Channel)
    output,x,y=net.GoogleNet()

    #动态调整learning rate
    global_step=tf.Variable(0,False)
    starter_learning_rate=0.1
    Learning_Rate=tf.train.exponential_decay(starter_learning_rate, global_step,
                                             10000, 0.9, staircase=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)+tf.add_n(tf.get_collection('loss')))

    # optimizer = tf.train.AdamOptimizer(learning_rate=Learning_Rate).minimize(loss,global_step=global_step)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=Learning_Rate).minimize(loss,global_step=global_step)

    y_test=tf.nn.softmax(output)
    correct_prediction=tf.equal(tf.argmax(y_test,1),tf.argmax(y,1))
    accurcy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #save model
    check_path='D:/Model/Stanford_Car/Stanford_Car_Google_Model.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #读入per_training model 
        var=tf.global_variables()
        var_to_restore=[val for val in var if 'linear_1' not in val.name]
        saver_pertrain=tf.train.Saver(var_to_restore)
        saver_pertrain.restore(sess,per_train_check_path)
        var_to_init=[val for val in var if 'linear_1' in val.name]
        tf.variables_initializer(var_to_init)

        # saver.save(sess, check_path)
        # saver.restore(sess, check_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        i=1
        while(True):
            train_images,train_labels=sess.run([train_x,train_y])

            # t=sess.run([pool],feed_dict={x:train_images,y:train_labels})
        
            sess.run([optimizer],feed_dict={x:train_images,y:train_labels})

            if((i%50)==0):
                train_images,train_labels=sess.run([train_x,train_y])
                train_loss,train_acc=sess.run([loss,accurcy],feed_dict={x:train_images,y:train_labels})
                print(' i=%d, train loss=%5f, train accurcy=%.5f'%(i,train_loss,train_acc))

            if ((i%500)==0):
                test_images,test_labels=sess.run([test_x,test_y])
                step,test_loss,test_acc=sess.run([global_step,loss,accurcy],feed_dict={x:test_images,y:test_labels})
                print('************ global_step=%d, Test loss=%5f, test accurcy=%.5f'%(step,test_loss,test_acc))

            if ((i%5000)==0):
                saver.save(sess, check_path)
            i+=1
        coord.request_stop()
        coord.join(threads)



if(__name__=='__main__'):
    train()

