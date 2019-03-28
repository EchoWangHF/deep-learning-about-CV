from mxnet.gluon import nn
from utils import *


def ConvBNBlock(channels, kernel_size, strides, pad, use_bias=False, leaky=True):
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(int(channels), kernel_size=kernel_size, strides=strides, padding=pad,
                      use_bias=use_bias))
    if not use_bias:
        blk.add(nn.BatchNorm(in_channels=int(channels)))
    if leaky:
        blk.add(nn.LeakyReLU(0.1))
    return blk


class ShortCutBlock(nn.HybridBlock):
    def __init__(self, channels):
        super(ShortCutBlock, self).__init__()
        self.conv_1 = ConvBNBlock(channels / 2, 1, 1, 0)
        self.conv_2 = ConvBNBlock(channels, 3, 1, 1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        blk = self.conv_1(x)
        blk = self.conv_2(blk)
        return blk + x


class UpSampleBlock(nn.HybridBlock):
    def __init__(self, scale, sample_type="nearest"):
        super(UpSampleBlock, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class TransformBlock(nn.HybridBlock):
    def __init__(self, num_classes, stride):
        super(TransformBlock, self).__init__()
        self.bbox_attrs = 5 + num_classes
        self.stride = stride

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.transpose(x.reshape((0, self.bbox_attrs * 3, self.stride * self.stride)), (0, 2, 1)).reshape((0, self.stride * self.stride * 3, self.bbox_attrs))
        xy_pred = F.sigmoid(x.slice_axis(begin=0, end=2, axis=-1))
        wh_pred = x.slice_axis(begin=2, end=4, axis=-1)
        score_pred = F.sigmoid(x.slice_axis(begin=4, end=5, axis=-1))
        cls_pred = F.sigmoid(x.slice_axis(begin=5, end=None, axis=-1))

        return F.concat(xy_pred, wh_pred, score_pred, cls_pred, dim=-1)


class DarkNet(nn.HybridBlock):
    def __init__(self, num_classes=80, input_dim=416):
        super(DarkNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        # self.anchors=my_anchors
        # self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
        #                 (59, 119), (116, 90), (156, 198), (373, 326)]

        self.conv_bn_block_0 = ConvBNBlock(32, 3, 1, 1) #32 416 416
        self.conv_bn_block_1 = ConvBNBlock(64, 3, 2, 1) #64 208 208
        self.shortcut_block_4 = ShortCutBlock(64)       #64 208 208
        self.conv_bn_block_5 = ConvBNBlock(128, 3, 2, 1)#128 104 104
        self.shortcut_block_8 = ShortCutBlock(128)      #128 104 104
        self.shortcut_block_11 = ShortCutBlock(128)     #128 104 104
        self.shortcut_block_14 = ShortCutBlock(128)
        self.shortcut_block_17 = ShortCutBlock(128)#128 104 104
        self.conv_bn_block_18 = ConvBNBlock(256, 3, 2, 1) #256 52 52 
        self.shortcut_block_21 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_24 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_27 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_30 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_33 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_36 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_39 = ShortCutBlock(256)      #256 52 52
        self.shortcut_block_42 = ShortCutBlock(256)#256 52 52
        self.conv_bn_block_43 = ConvBNBlock(512, 3, 2, 1)#512 26 26
        self.shortcut_block_46 = ShortCutBlock(512)
        self.shortcut_block_49 = ShortCutBlock(512)
        self.shortcut_block_52 = ShortCutBlock(512)
        self.shortcut_block_55 = ShortCutBlock(512)
        self.shortcut_block_58 = ShortCutBlock(512)
        self.shortcut_block_61 = ShortCutBlock(512)
        self.shortcut_block_64 = ShortCutBlock(512)
        self.shortcut_block_67 = ShortCutBlock(512)#512 26 26
        self.conv_bn_block_68 = ConvBNBlock(1024, 3, 2, 1)#1024 13 13
        self.shortcut_block_71 = ShortCutBlock(1024)
        self.shortcut_block_74 = ShortCutBlock(1024)
        self.shortcut_block_77 = ShortCutBlock(1024)
        self.shortcut_block_80 = ShortCutBlock(1024)#1024 13 13
        self.conv_bn_block_81 = ConvBNBlock(512, 1, 1, 0)
        self.conv_bn_block_82 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_83 = ConvBNBlock(512, 1, 1, 0)
        self.conv_bn_block_84 = ConvBNBlock(1024, 3, 1, 1)# 1024 13 13
        self.conv_bn_block_85 = ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False) # * 13 13
        self.transform_0 = TransformBlock(num_classes, 13)

        self.conv_bn_block_87=ConvBNBlock(256,1,1,0)
        self.upsample_block_88=UpSampleBlock(scale=2)
        self.conv_bn_block_89=ConvBNBlock(256,1,1,0)
        self.conv_bn_block_90=ConvBNBlock(512,3,1,1)
        self.conv_bn_block_91=ConvBNBlock(256,1,1,0)
        self.conv_bn_block_92=ConvBNBlock(512,3,1,1)
        self.conv_bn_block_93=ConvBNBlock(256,1,1,0)
        self.conv_bn_block_94=ConvBNBlock(512,3,1,1)
        self.conv_bn_block_95=ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_1=TransformBlock(num_classes,26)

        self.conv_bn_block_96=ConvBNBlock(128,1,1,0)
        self.upsample_block_97=UpSampleBlock(scale=2)
        self.conv_bn_block_98=ConvBNBlock(128,1,1,0)
        self.conv_bn_block_99=ConvBNBlock(256,3,1,1)
        self.conv_bn_block_100=ConvBNBlock(128,1,1,0)
        self.conv_bn_block_101=ConvBNBlock(256,3,1,1)
        self.conv_bn_block_102=ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_2=TransformBlock(num_classes,52)

        self.conv_bn_block_103=ConvBNBlock(64,1,1,0)
        self.upsample_block_104=UpSampleBlock(scale=2)
        self.conv_bn_block_105=ConvBNBlock(64,1,1,0)
        self.conv_bn_block_106=ConvBNBlock(128,1,1,0)
        self.conv_bn_block_107=ConvBNBlock(64,1,1,0)
        self.conv_bn_block_108=ConvBNBlock(128,1,1,0)
        self.conv_bn_block_109=ConvBNBlock(3 * (5+self.num_classes), 1, 1, 0, use_bias=True, leaky=False)
        self.transform_3=TransformBlock(num_classes,104)



    def hybrid_forward(self, F, x, *args, **kwargs):
        x=self.conv_bn_block_0(x)
        x=self.conv_bn_block_1(x)
        x=self.shortcut_block_4(x)
        x=self.conv_bn_block_5(x)
        x=self.shortcut_block_8(x)
        x=self.shortcut_block_11(x)
        x=self.shortcut_block_14(x)
        shortcut_17=self.shortcut_block_17(x)
        x=self.conv_bn_block_18(shortcut_17)
        x=self.shortcut_block_21(x)
        x=self.shortcut_block_24(x)
        x=self.shortcut_block_27(x)
        x=self.shortcut_block_30(x)
        x=self.shortcut_block_33(x)
        x=self.shortcut_block_36(x)
        x=self.shortcut_block_39(x)
        shortcut_42=self.shortcut_block_42(x)
        x=self.conv_bn_block_43(shortcut_42)
        x=self.shortcut_block_46(x)
        x=self.shortcut_block_49(x)
        x=self.shortcut_block_52(x)
        x=self.shortcut_block_55(x)
        x=self.shortcut_block_58(x)
        x=self.shortcut_block_61(x)
        x=self.shortcut_block_64(x)
        shortcut_67=self.shortcut_block_67(x)
        x=self.conv_bn_block_68(shortcut_67)
        x=self.shortcut_block_71(x)
        x=self.shortcut_block_74(x)
        x=self.shortcut_block_77(x)
        x=self.shortcut_block_80(x)
        x=self.conv_bn_block_81(x)
        x=self.conv_bn_block_82(x)
        conv_83=self.conv_bn_block_83(x)
        x=self.conv_bn_block_84(x)
        x=self.conv_bn_block_85(x)

        predict_0=self.transform_0(x)

        route_0=conv_83
        conv_87=self.conv_bn_block_87(route_0)
        upsample_88=self.upsample_block_88(conv_87)
        route_1=F.concat(upsample_88,shortcut_67,dim=1)
        x=self.conv_bn_block_89(route_1)
        x=self.conv_bn_block_90(x)
        x=self.conv_bn_block_91(x)
        x=self.conv_bn_block_92(x)
        conv_93=self.conv_bn_block_93(x)
        x=self.conv_bn_block_94(conv_93)
        x=self.conv_bn_block_95(x)

        predict_1=self.transform_1(x)

        route_2=conv_93
        conv_96=self.conv_bn_block_96(route_2)
        upsample_97=self.upsample_block_97(conv_96)
        route_3=F.concat(upsample_97,shortcut_42,dim=1)
        x=self.conv_bn_block_98(route_3)
        x=self.conv_bn_block_99(x)
        conv_100=self.conv_bn_block_100(x)
        x=self.conv_bn_block_101(conv_100)
        x=self.conv_bn_block_102(x)

        predict_2=self.transform_2(x)

        route_4=conv_100
        conv_103=self.conv_bn_block_103(route_4)
        upsample_104=self.upsample_block_104(conv_103)
        route_5=F.concat(upsample_104,shortcut_17,dim=1)
        x=self.conv_bn_block_105(route_5)
        x=self.conv_bn_block_106(x)
        x=self.conv_bn_block_107(x)
        x=self.conv_bn_block_108(x)
        x=self.conv_bn_block_109(x)

        predict_3=self.transform_3(x)

        detections=F.concat(predict_0,predict_1,predict_2,predict_3,dim=1)
        return detections

    def load_weights(self, weightfile, fine_tune):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]

        weights = nd.array(np.fromfile(fp, dtype=np.float32))
        ptr = 0

        def set_data(model, ptr):
            conv = model[0]
            if len(model) > 1:
                bn = model[1]

                # Get the number of weights of Batch Norm Layer
                num_bn_beta = self.numel(bn.beta.shape)
                # Load the weights
                bn_beta = weights[ptr:ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_gamma = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_running_mean = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_running_var = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                # Cast the loaded weights into dims of model weights.
                bn_beta = bn_beta.reshape(bn.beta.shape)
                bn_gamma = bn_gamma.reshape(bn.gamma.shape)
                bn_running_mean = bn_running_mean.reshape(bn.running_mean.shape)
                bn_running_var = bn_running_var.reshape(bn.running_var.shape)

                bn.gamma.set_data(bn_gamma)
                bn.beta.set_data(bn_beta)
                bn.running_mean.set_data(bn_running_mean)
                bn.running_var.set_data(bn_running_var)
            else:
                num_biases = self.numel(conv.bias.shape)

                conv_biases = weights[ptr: ptr + num_biases]
                ptr = ptr + num_biases

                conv_biases = conv_biases.reshape(conv.bias.shape)

                conv.bias.set_data(conv_biases)

            num_weights = self.numel(conv.weight.shape)

            conv_weights = weights[ptr:ptr + num_weights]
            ptr = ptr + num_weights

            conv_weights = conv_weights.reshape(conv.weight.shape)

            conv.weight.set_data(conv_weights)
            return ptr

        modules = self._children
        for block_name in modules:
            if fine_tune:
                if block_name.find("81") != -1:
                    ptr = 56629087
                    continue
                elif block_name.find("93") != -1:
                    ptr = 60898910
                    continue
                elif block_name.find("105") != -1:
                    continue
            module = modules.get(block_name)
            if isinstance(module, nn.HybridSequential):
                ptr = set_data(module, ptr)
            elif isinstance(module, ShortCutBlock):
                shortcut_modules = module._children
                for shortcut_name in shortcut_modules:
                    ptr = set_data(shortcut_modules.get(shortcut_name), ptr)
            elif isinstance(module, UpSampleBlock) or isinstance(module, TransformBlock):
                continue
            else:
                print(module)
                print("load weights wrong")

    def numel(self, x):
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        return np.prod(x)


class TinyDarkNet(nn.HybridBlock):
    def __init__(self, num_classes=80, input_dim=416):
        super(TinyDarkNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]

        self.conv_bn_block_0 = ConvBNBlock(16, 3, 1, 1)
        self.max_pool_1 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_2 = ConvBNBlock(32, 3, 1, 1)
        self.max_pool_3 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_4 = ConvBNBlock(64, 3, 1, 1)
        self.max_pool_5 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_6 = ConvBNBlock(128, 3, 1, 1)
        self.max_pool_7 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_8 = ConvBNBlock(256, 3, 1, 1)
        self.max_pool_9 = nn.MaxPool2D(2, 2)
        self.conv_bn_block_10 = ConvBNBlock(512, 3, 1, 1)
        self.max_pool_11 = nn.MaxPool2D(2, 1,ceil_mode=True)
        self.conv_bn_block_12 = ConvBNBlock(1024, 3, 1, 1)
        self.conv_bn_block_13 = ConvBNBlock(256, 1, 1, 1)
        self.conv_bn_block_14 = ConvBNBlock(512, 3, 1, 1)
        self.conv_bn_block_15 = ConvBNBlock(3 * (5 + self.num_classes), 1, 1, 1, use_bias=True, leaky=False)
        self.transform_0 = TransformBlock(self.num_classes, 13)
        self.conv_bn_block_16 = ConvBNBlock(128, 1, 1, 1)
        self.upsample_block_17 = UpSampleBlock(scale=2)
        self.conv_bn_block_18 = ConvBNBlock(256, 3, 1, 1)
        self.conv_bn_block_19 = ConvBNBlock(3 * (5 + self.num_classes), 1, 1, 1, use_bias=True, leaky=False)
        self.transform_1 = TransformBlock(self.num_classes, 26)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv_bn_block_0(x)
        x = self.max_pool_1(x)
        x = self.conv_bn_block_2(x)
        x = self.max_pool_3(x)
        x = self.conv_bn_block_4(x)
        x = self.max_pool_5(x)
        x = self.conv_bn_block_6(x)
        x = self.max_pool_7(x)
        conv_8 = self.conv_bn_block_8(x)
        x = self.max_pool_9(conv_8)
        x = self.conv_bn_block_10(x)
        x = self.max_pool_11(x)
        x = self.conv_bn_block_12(x)
        conv_13 = self.conv_bn_block_13(x)
        x = self.conv_bn_block_14(conv_13)
        conv_15 = self.conv_bn_block_15(x)

        predict_16 = self.transform_0(conv_15)

        x = self.conv_bn_block_16(conv_13.copy())
        x = self.upsample_block_17(x)
        x = nd.concat(x, conv_8, dim=1)
        x = self.conv_bn_block_18(x)
        x = self.conv_bn_block_19(x)
        predict_20 = self.transform_1(x)
        detections = nd.concat(predict_16, predict_20, dim=1)

        return detections

    def load_weights(self, weightfile, fine_tune):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]

        weights = nd.array(np.fromfile(fp, dtype=np.float32))
        ptr = 0

        def set_data(model, ptr):
            conv = model[0]
            if len(model) > 1:
                bn = model[1]

                # Get the number of weights of Batch Norm Layer
                num_bn_beta = self.numel(bn.beta.shape)
                # Load the weights
                bn_beta = weights[ptr:ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_gamma = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_running_mean = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                bn_running_var = weights[ptr: ptr + num_bn_beta]
                ptr += num_bn_beta

                # Cast the loaded weights into dims of model weights.
                bn_beta = bn_beta.reshape(bn.beta.shape)
                bn_gamma = bn_gamma.reshape(bn.gamma.shape)
                bn_running_mean = bn_running_mean.reshape(bn.running_mean.shape)
                bn_running_var = bn_running_var.reshape(bn.running_var.shape)

                bn.gamma.set_data(bn_gamma)
                bn.beta.set_data(bn_beta)
                bn.running_mean.set_data(bn_running_mean)
                bn.running_var.set_data(bn_running_var)
            else:
                num_biases = self.numel(conv.bias.shape)

                conv_biases = weights[ptr: ptr + num_biases]
                ptr = ptr + num_biases

                conv_biases = conv_biases.reshape(conv.bias.shape)

                conv.bias.set_data(conv_biases)

            num_weights = self.numel(conv.weight.shape)

            conv_weights = weights[ptr:ptr + num_weights]
            ptr = ptr + num_weights

            conv_weights = conv_weights.reshape(conv.weight.shape)

            conv.weight.set_data(conv_weights)
            return ptr

        modules = self._children
        for block_name in modules:
            if fine_tune:
                if block_name.find("81") != -1:
                    ptr = 56629087
                    continue
                elif block_name.find("93") != -1:
                    ptr = 60898910
                    continue
                elif block_name.find("105") != -1:
                    continue
            module = modules.get(block_name)
            if isinstance(module, nn.HybridSequential):
                ptr = set_data(module, ptr)
            elif isinstance(module, UpSampleBlock) or isinstance(module, TransformBlock) or isinstance(module, nn.MaxPool2D):
                continue
            else:
                print(module)
                print("load weights wrong")

    def numel(self, x):
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        return np.prod(x)


if __name__ == '__main__':
    net = DarkNet()
    net.initialize(init=mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    X = nd.uniform(shape=(1,3, 416, 416))
    detections = net(X)
    # darknet.load_weights("yolov3.weights")
    print(detections.shape)
