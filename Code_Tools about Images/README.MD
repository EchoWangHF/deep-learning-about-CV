
# 一些代码工具的打包总结

#### 1 windows 环境下遍历文件夹 [here](https://github.com/EchoWangHF/Learning-mxnet/blob/master/Code_Tools%20about%20Images/traverse_folder.cpp)

#### 2 mxnet 如何制作自己的图片数据分类算法训练集
1.mxnet 图片训练集需要生成.rec 文件，官方教程：[here](https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html)</br>
基本步骤为：</br>
(1)基于原图片生成lst文件。如果源图片文件存在example文件夹当中，并且这些源图片文件已经按照类别存放在不同的文件当中。</br>
`python im2rec.py ./example_rec ./example/ --recursive --list --num-thread 2`</br>
可以在当中加入其他参数，如：`--resize 64` 调节尺寸大小，`--color 1`调节图片的是否为彩色还是灰度,具体参数可以看源码。</br>
(2)基于生成的.lst文件生成rec文件。</br>
`python im2rec.py ./example_rec ./example/ --recursive --pass-through --pack-label --num-thread 2`</br>
此时，则会生成.idx和.rec文件，则可以用作作为训练集。</br>

2. 使用`gluon.data.DataLoader()`制作数据集，前提都是将数据按照类别，用不同的文件夹分好类。代码可见：example
