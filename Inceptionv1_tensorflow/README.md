# Inception_v1 (GoogleNet)


I use the ImageNet to training the 'GoogleNet' with tensorflow. The per-training  model can download from [here](https://pan.baidu.com/s/1d8QlKi_3EUyNbRGxnQg1YQ).</br>

The `GoogleNet` is written in the `Net.py`. Moreover, the `VGG16` is also included in `Net.py`</br>

The parameters of Net such as batch_size and class_num can be set in the `Param.py`</br>

`TFRecord.py` include the method that how to create and decode the TFRecord File with tensorflow.</br>

You can change the class_num in Param.py and use the `per_training_main.py` to accurate your model training.</br>
