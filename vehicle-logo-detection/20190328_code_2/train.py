import argparse
import os
import re
import time
from random import shuffle
from mxnet import autograd
from darknet import DarkNet
from utils import *
import loss


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images_path', default="/media/seeking/新加卷/dataset/VL1/",type=str)
    parser.add_argument("--train", dest='train_data_path', type=str)
    parser.add_argument("--val", dest='val_data_path', type=str)
    parser.add_argument("--coco_train", dest="coco_train", type=str)
    parser.add_argument("--coco_val", dest="coco_val", type=str)
    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--classes", dest="classes", default="my_mxnet_pro_2/data/vcc.names", type=str)
    parser.add_argument("--prefix", dest="prefix", default="voc")
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default='0', type=str)
    parser.add_argument("--dst_dir", dest='dst_dir', default="results", type=str)
    parser.add_argument("--epoch", dest="epoch", default=500, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=10, type=int)
    parser.add_argument("--ignore_thresh", dest="ignore_thresh", default=0.6)
    parser.add_argument("--params", dest='params', help=
    "mxnet params file", default="/home/seeking/Vehicle_Data/models/yolov3.weights", type=str)
    parser.add_argument("--input_dim", dest='input_dim', default=416, type=int)

    return parser.parse_args()


def calculate_ignore(prediction, true_xywhs, ignore_thresh):
    ctx = prediction.context
    tmp_pred = predict_transform(prediction, input_dim, anchors)
    ignore_mask = nd.ones(shape=pred_score.shape, dtype="float32", ctx=ctx)
    item_index = np.argwhere(true_xywhs[:, :, 4].asnumpy() == 1.0)

    for x_box, y_box in item_index:
        iou = bbox_iou(tmp_pred[x_box, y_box:y_box + 1, :4], true_xywhs[x_box, y_box:y_box + 1])
        ignore_mask[x_box, y_box] = (iou < ignore_thresh).astype("float32").reshape(-1)
    return ignore_mask


class YoloDataSet(gluon.data.Dataset):
    def __init__(self, data_path, classes, input_dim=416, is_shuffle=False, mode="train"):
        super(YoloDataSet, self).__init__()
        self.anchors=my_anchors
        self.classes=classes
        self.input_dim = input_dim
        self.label_mode = "xml"
        self.label_list = []
        self.image_list=[]
        if os.path.isdir(data_path):
            self.label_path=os.path.join(data_path,'annotation')
            self.image_path=os.path.join(data_path,mode)

            length=len(classes)
            for each_class_index in range(length):
                each_class=classes[each_class_index]
                image_path_temp=os.path.join(self.image_path,each_class)
                image_names=os.listdir(image_path_temp)

                for each_image_name in image_names:
                    image_=os.path.join(image_path_temp,each_image_name.strip())
                    if(os.path.isfile(image_)==False):
                        continue
                    else:
                        self.image_list.append(image_)

                    label_temp=[]
                    label_=os.path.join(self.label_path,each_image_name.strip()+'.xml')
                    if(os.path.isfile(label_)==False):
                        continue
                    else:
                        label_temp.append(label_)
                        label_temp.append(each_class_index)
                        self.label_list.append(label_temp)


            # self.label_path=os.path.join(data_path,'annotation')

            # for i in image_name:
            #     image_=os.path.join(self.image_path,i.strip())
            #     if(os.path.isfile(image_)==False):
            #         continue
            #     else:
            #         label_=os.path.join(self.label_path,i.strip()+'.xml')
            #         if(os.path.isfile(label_)==False):
            #             continue
            #         else:
            #             self.image_list.append(image_)
            #             self.label_list.append(label_)
        
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        label = prep_label_new(self.label_list[idx], classes=self.classes)
        image, label = prep_image(image, self.input_dim, label)
        label, true_xywhc = prep_final_label(label, len(self.classes), input_dim=self.input_dim)
        return nd.array(image).squeeze(), label.squeeze(), true_xywhc.squeeze()


if __name__ == '__main__':
    args = arg_parse()
    if args.images_path:
        args.train_data_path = args.images_path
        args.val_data_path = args.images_path
    classes = load_classes(args.classes)
    num_classes = len(classes)
    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(gpu)
    input_dim = args.input_dim
    batch_size = args.batch_size

    train_dataset = YoloDataSet(args.train_data_path, classes=classes, is_shuffle=True, mode="train")

    train_dataloader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        "train": train_dataloader
    }
    if args.val_data_path:
        val_dataset = YoloDataSet(args.val_data_path, classes=classes, is_shuffle=True, mode="val")
        val_dataloader = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        dataloaders["val"] = val_dataloader

    obj_loss = LossRecorder('objectness_loss')
    cls_loss = LossRecorder('classification_loss')
    box_loss = LossRecorder('box_refine_loss')
    positive_weight = 1.0
    negative_weight = 1.0

    l2_loss = L2Loss(weight=2.)
    focal_loss=FocalLoss()

    net = DarkNet(num_classes=num_classes, input_dim=input_dim)
    net.initialize(init=mx.init.Xavier(), ctx=ctx)
    # print(net)
    if args.params.endswith(".params"):
        net.load_parameters(args.params)
    elif args.params.endswith(".weights"):
        X = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx=ctx[-1])
        net(X)
        net.load_weights(args.params, fine_tune=num_classes)
    else:
        print("params {} load error!".format(args.params))
        exit()
    print("load params: {}".format(args.params))
    net.hybridize()
    
    # anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        # (59, 119), (116, 90), (156, 198), (373, 326)])
    anchors=np.array(my_anchors)

    total_steps = int(np.ceil(len(train_dataset) / batch_size) - 1)
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=[200 * total_steps], factor=0.1)
    # optimizer = mx.optimizer.Adam(learning_rate=args.lr, lr_scheduler=schedule)
    optimizer = mx.optimizer.SGD(learning_rate=args.lr, lr_scheduler=schedule,momentum=0.9)
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

    best_loss = 100000.
    early_stop = 0

    for epoch in range(500):
        # if early_stop >= 5:
        #     print("train stop, epoch: {0}  best loss: {1:.3f}".format(epoch - 5, best_loss))
        #     break
        print('Epoch {} / {}'.format(epoch, 500 - 1))

        for mode in ["train", "val"]:
            tic = time.time()
            if mode == "val":
                if not args.val_data_path:
                    continue
                total_steps = int(np.ceil(len(val_dataset) / batch_size) - 1)
            else:
                total_steps = int(np.ceil(len(train_dataset) / batch_size) - 1)
            cls_loss.reset()
            obj_loss.reset()
            box_loss.reset()
            for i, batch in enumerate(dataloaders[mode]):
                gpu_Xs = split_and_load(batch[0], ctx)
                gpu_Ys = split_and_load(batch[1], ctx)
                gpu_Zs = split_and_load(batch[2], ctx)
                with autograd.record(mode == "train"):
                    loss_list = []
                    batch_num = 0
                    for gpu_x, gpu_y, gpu_z in zip(gpu_Xs, gpu_Ys, gpu_Zs):
                        mini_batch_size = gpu_x.shape[0]
                        prediction = net(gpu_x)
                        # print(prediction.shape)
                        pred_xywh = prediction[:, :, :4]
                        pred_score = prediction[:, :, 4:5]
                        pred_cls = prediction[:, :, 5:]
                        with autograd.pause():
                            ignore_mask = calculate_ignore(prediction.copy(), gpu_z, args.ignore_thresh)
                            true_box = gpu_y[:, :, :4]
                            true_score = gpu_y[:, :, 4:5]
                            true_cls = gpu_y[:, :, 5:]
                            coordinate_weight = true_score.copy()
                            score_weight = nd.where(coordinate_weight == 1.0,
                                                    nd.ones_like(coordinate_weight) * positive_weight,
                                                    nd.ones_like(coordinate_weight) * negative_weight)
                            box_loss_scale = 2. - gpu_z[:, :, 2:3] * gpu_z[:, :, 3:4] / float(args.input_dim ** 2)
                    
                        loss_xywh = l2_loss(pred_xywh, true_box,ignore_mask * coordinate_weight * box_loss_scale)

                        loss_conf = l2_loss(pred_score, true_score)

                        loss_cls = focal_loss(pred_cls, true_cls,)

                        t_loss_xywh = nd.sum(loss_xywh) / mini_batch_size

                        t_loss_conf = nd.sum(loss_conf) / mini_batch_size

                        t_loss_cls = nd.sum(loss_cls) / mini_batch_size

                        loss = t_loss_xywh + t_loss_conf + t_loss_cls
                        batch_num += len(loss)

                        if mode == "train":
                            loss.backward()
                        with autograd.pause():
                            loss_list.append(loss.asscalar())
                            cls_loss.update([t_loss_cls])
                            obj_loss.update([t_loss_conf])
                            box_loss.update([t_loss_xywh])

                trainer.step(batch_num, ignore_stale_grad=True)

                if (i + 1) % 20 == 0:
                    mean_loss = 0.
                    for l in loss_list:
                        mean_loss += l
                    mean_loss /= len(loss_list)
                    print("{0}  epoch: {1}  batch: {2} / {3}  loss: {4:.3f}".format(mode, epoch, i, total_steps, mean_loss))
                
                # if (i + 1) % int(total_steps / 2) == 0:
                #     total_num = nd.sum(coordinate_weight)
                #     item_index = np.nonzero(true_score.asnumpy())
                #     print("predict case / right case: {}".format((nd.sum(pred_score > 0.5) / total_num).asscalar()))
                #     print((nd.sum(nd.abs(pred_score * coordinate_weight - true_sc  ore)) / total_num).asscalar())
            nd.waitall()
            # print('Epoch %2d, %s %s %.5f, %s %.5f, %s %.5f time %.1f sec' % (
            #       epoch, mode, *cls_loss.get(), *obj_loss.get(), *box_loss.get(), time.time() - tic))
        loss = cls_loss.get()[1] + obj_loss.get()[1] + box_loss.get()[1]
        print('sum_loss: {0}'.format(loss))

        net.save_parameters("/home/seeking/Vehicle_Data/models/logo_class_yolov3_mxnet.params")
        # if loss < best_loss:
        #     early_stop = 0
        #     best_loss = loss
        #     # net.save_params("/home/seeking/Vehicle_Data/models/{0}_yolov3_mxnet.params".format(args.prefix))
        #     net.save_parameters("/home/seeking/Vehicle_Data/models/"+str(epoch)+"_yolov3_mxnet.params")
        # else:
        #     early_stop += 1


