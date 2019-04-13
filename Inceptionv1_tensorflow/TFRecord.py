import tensorflow as tf
import numpy as np
import PIL.Image as Image
import os
import Param 
import cv2


def shuffle_train_file():
    train_image_path='F:\\ImageNet\\train\\'
    Image_List=[]
    for c in range(0,569,1): #569
        sample=0
        while(sample<1500):
            list_temp=[]
            image_path=train_image_path+str(c)+'_'+str(sample)+'.JPEG'
            if not os.path.exists(image_path):
                # print(image_path)
                sample+=1
                continue
            list_temp.append(image_path)
            list_temp.append(str(c))
            sample+=1
            Image_List.append(list_temp)
    
    Image_Array=np.array(Image_List,dtype='str')
    np.random.shuffle(Image_Array)
    return Image_Array

def Create_Train_TFRecord():
    num_shards=58
    per_shards=5000

    Image_Array=shuffle_train_file()

    length=len(Image_Array)

    for i in range(0,num_shards,1):
        filename=('D:/data_set/ImageNet/train.tfrecords-%.2d-of-%.3d'%(i,num_shards))
        writer=tf.python_io.TFRecordWriter(filename)
        for j in range(0,per_shards,1):
            tag=i*per_shards+j
            if(tag>=length):
                print('out of number!')
                return
            each=Image_Array[tag]

            image_path=str(each[0])
            index=int(each[1])

            img=Image.open(image_path)
            if(img.mode!='RGB'):
                print('Not RGB   ',image_path)
                continue
        
            if((img.size[0]<64)or(img.size[1]<64)):
                print('Size is too Small  ',image_path)
                continue

            img=img.resize((Param.Width,Param.Height))
            img_raw=img.tobytes()

            example=tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
            writer.write(example.SerializeToString())
            if(tag%500)==0:
                print(i,'  ',j,' ',tag)
        writer.close() 

def Create_Val_TFRecord():
    val_image_path='F:\\ImageNet\\val\\'
    filename='D:/data_set/ImageNet/val.tfrecords'
    writer=tf.python_io.TFRecordWriter(filename)
    for each in os.listdir(val_image_path):
        class_tag,_=each.split('n')
        image_path=val_image_path+each
        val_index=int(class_tag)

        img=Image.open(image_path)
        if(img.mode!='RGB'):
            print('Not RGB   ',image_path)
            continue
        
        if((img.size[0]<64)or(img.size[1]<64)):
            print('Size is too Small  ',image_path)
            continue
        img=img.resize((Param.Width,Param.Height))
        img_raw=img.tobytes()

        example=tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(val_index)])),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())
    writer.close()

def Read_Train_TFRecord():
    # files=tf.train.match_filenames_once('D:/data_set/ImageNet/train.tfrecords-*')
    TFRecord_file=[os.path.join('D:/data_set/ImageNet/','train.tfrecords-%.2d-of-058'% i) for i in range(0,58)]
    filename_queue=tf.train.string_input_producer(TFRecord_file,num_epochs=Param.Epoch)
    reader=tf.TFRecordReader()
    _,example=reader.read(filename_queue)
    features=tf.parse_single_example(example,features={'label':tf.FixedLenFeature([],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
    images=tf.decode_raw(features['img_raw'],tf.uint8)
    images=tf.reshape(images,[Param.Width,Param.Height,Param.Channel])
    labels=tf.cast(features['label'],tf.int64)

    images, labels=tf.train.shuffle_batch([images,labels],batch_size=Param.Batch,capacity=16*Param.Batch,min_after_dequeue=4*Param.Batch,num_threads=Param.Threads_num)

    labels = tf.one_hot(labels, Param.Class_Num)

    print("Train  read over!")
    return images,labels

def Read_Val_TFRecord():
    TFRecord_file='D:/data_set/ImageNet/val.tfrecords'
    if not os.path.exists(TFRecord_file):
        Create_Val_TFRecord()
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([TFRecord_file],num_epochs=Param.Epoch)
    _,example=reader.read(filename_queue)
    features=tf.parse_single_example(example,features={'label':tf.FixedLenFeature([],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
    images=tf.decode_raw(features['img_raw'],tf.uint8)
    images=tf.reshape(images,[Param.Width,Param.Height,Param.Channel])

    labels=tf.cast(features['label'],tf.int64)

    images, labels=tf.train.shuffle_batch([images,labels],batch_size=Param.Batch,capacity=16*Param.Batch,min_after_dequeue=4*Param.Batch,num_threads=Param.Threads_num)

    labels = tf.one_hot(labels, Param.Class_Num)

    print("val read over!")
    return images,labels

if __name__=='__main__':
    # Create_Train_TFRecord()
    # Create_Val_TFRecord
    train_x,train_y=Read_Train_TFRecord()
    # train_x,train_y=Read_Val_TFRecord()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for t in range(0,5,1):
            train_images, train_labels = sess.run([train_x, train_y])
            print(train_labels)
            for i in range(Param.Batch):
                img=np.reshape(train_images[i],[Param.Width,Param.Height,Param.Channel])
                img=Image.fromarray(img,'RGB')
                img.save('E:\\'+ str(i) +'+'+ str(train_labels[i]) + '.jpg')
    coord.request_stop()
    coord.join(threads)

            
