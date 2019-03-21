import os
import time
import argparse
from utils import *
from backbone import DarkNet, TinyDarkNet
import numpy as np
import gluoncv

image_name = 0


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images',help=
    "Image / Directory containing images to perform detection upon",
                        default="/home/seeking/Vehicle_Data/data/test", type=str)
    parser.add_argument("--video", dest='video', help=
    "video file path", type=str)
    parser.add_argument("--classes", dest="classes", default="my_mxnet_pro/data/vcc.names", type=str)
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default="0", type=str)
    parser.add_argument("--dst_dir", dest='dst_dir', help=
    "Image / Directory to store detections to", default="/home/seeking/Vehicle_Data/data/save", type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=30, type=int)
    parser.add_argument("--tiny", dest="tiny", help="use yolov3-tiny", default=False, type=bool)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence", default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.5, type=float)
    parser.add_argument("--params", dest='params', help=
    "params file", default="/home/seeking/Vehicle_Data/models/logo_focalloss_yolov3_mxnet.params", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)

    return parser.parse_args()


def draw_bbox(file_path,img, bboxs):
    args = arg_parse()
    dst_path=args.dst_dir
    image_name=file_path.split('/')[-1]
    image_path=os.path.join(dst_path,image_name)
    
    c1 = tuple([bboxs[0].asscalar().astype("int"),bboxs[1].asscalar().astype("int")])
    c2 = tuple([bboxs[2].asscalar().astype("int"),bboxs[3].asscalar().astype("int")])
    color = (0, 0, 255)
    cv2.rectangle(img, c1, c2, color, 1)
    cv2.imwrite(image_path,img)



def iou_reco(file_path,bboxs):

    file_path=file_path.split('/')[-1]
    gt_bbox=prase__annotation_xml(file_path)

    pre_bbox=bboxs[:4]
    
    pre_bbox=pre_bbox.reshape([1,-1])
    
    # iou_temp=my_bbox_iou(gt_bbox,pre_bbox)
    iou_temp=bbox_iou(gt_bbox,pre_bbox,False)
    if iou_temp>0.5:
        return True
    else:
        return False

def cal_iou(imagefile,load_images,output, input_dim):
    
    print('output shape: ',output.shape)

    num_true=0.0

    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images],ctx=mx.gpu())

    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 4], :]

    print('im_dim_list shape: ',im_dim_list.shape)

    scaling_factor = nd.min(input_dim / im_dim_list, axis=1).reshape((-1, 1))

    output[:, [0, 2]] -= (input_dim - scaling_factor * im_dim_list[:, 0].reshape((-1, 1))) / 2
    output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim_list[:, 1].reshape((-1, 1))) / 2
    output[:, 0:4] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [0, 2]] = nd.clip(output[i, [0, 2]], a_min=0.0, a_max=im_dim_list[i][0].asscalar())
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i][1].asscalar())
    
    num_=output.shape[0]
    for i in range(num_):
        res_temp=output[i,:4]
        index=int(output[i,4].asscalar())
        if(iou_reco(imagefile[index],res_temp)):
            num_true+=1
    return num_true

def prase__annotation_xml(image_name):
    # return class_name,BBox
    annotation_xml_path="/home/seeking/Vehicle_Data/data/annotation/"
    annotation_xml_name=annotation_xml_path+image_name+'.xml'

    root=ET.parse(annotation_xml_name).getroot()
    bbox=[]
    start_x=0
    start_y=0
    image_width=0
    image_hight=0
    for i in root:
        name=str(i.tag)

        if(name=='车'):
            box_list=[]
            for each in i:
                box_list.append(float(each.text))
            start_x=box_list[0]
            start_y=box_list[1]
            image_width=box_list[2]
            image_hight=box_list[3]
        else:
            if(name=='车标'):
                box_list=[]
                for each in i:
                    box_list.append(float(each.text))
                
                assert(len(box_list)==4)
                xmin=box_list[0]-start_x
                ymin=box_list[1]-start_y
                xmax=xmin+box_list[2]
                ymax=ymin+box_list[3]
                
                bbox.append([xmin,ymin,xmax,ymax])
                
    return nd.array(bbox,ctx=mx.gpu())

def draw_images(batch,load_images,output,input_dim):
    print('output shape: ',output.shape)

    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images],ctx=mx.gpu())

    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 4], :]

    print('im_dim_list shape: ',im_dim_list.shape)

    scaling_factor = nd.min(input_dim / im_dim_list, axis=1).reshape((-1, 1))

    output[:, [0, 2]] -= (input_dim - scaling_factor * im_dim_list[:, 0].reshape((-1, 1))) / 2
    output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim_list[:, 1].reshape((-1, 1))) / 2
    output[:, 0:4] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [0, 2]] = nd.clip(output[i, [0, 2]], a_min=0.0, a_max=im_dim_list[i][0].asscalar())
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i][1].asscalar())
    
    num_=output.shape[0]
    for i in range(num_):
        res_temp=output[i,:4]
        index=int(output[i,4].asscalar())
        draw_bbox(batch[index],load_images[index],res_temp)


def detect_mAP():
    # 计算iou 精确度
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    input_dim = args.input_dim

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    
    net = DarkNet(input_dim=input_dim)

    anchors=my_anchors
    net.initialize(ctx=ctx)

    try:
        imlist = [os.path.join(images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))

    if args.params.endswith(".params"):
        net.load_parameters(args.params)
    elif args.params.endswith(".weights"):
        tmp_batch = nd.uniform(shape=(1, 3, args.input_dim, args.input_dim), ctx=ctx)
        net(tmp_batch)
        net.load_weights(args.params, fine_tune=False)
    else:
        print("params {} load error!".format(args.params))
        exit()
    print("load params: {}".format(args.params))
    net.hybridize()

    if not imlist:
        print("no images to detect")
        exit()
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1

    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
                  for i in range(num_batches)]

    num_true=0.0
    num_samples=0.0
    for i, batch in enumerate(im_batches):
        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        prediction = my_NMS(prediction)
        
        num_samples=num_samples+len(batch)
        if prediction is not None:
            num_true_temp=cal_iou(batch,load_images,prediction, input_dim=input_dim)
            num_true=num_true+num_true_temp
            print(i)
        else:
            print("No detections were made")
    
    print(num_true)
    print(num_true/num_samples)

def detect_draw():
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    input_dim = args.input_dim
    dst_dir=args.dst_dir

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    
    net = DarkNet(input_dim=input_dim)

    anchors=my_anchors
    net.initialize(ctx=ctx)

    try:
        imlist = [os.path.join(images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    if args.params.endswith(".params"):
        net.load_parameters(args.params)
    elif args.params.endswith(".weights"):
        tmp_batch = nd.uniform(shape=(1, 3, args.input_dim, args.input_dim), ctx=ctx)
        net(tmp_batch)
        net.load_weights(args.params, fine_tune=False)
    else:
        print("params {} load error!".format(args.params))
        exit()
    print("load params: {}".format(args.params))
    net.hybridize()

    if not imlist:
        print("no images to detect")
        exit()
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1

    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
                  for i in range(num_batches)]

    for i, batch in enumerate(im_batches):
        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)
        start = time.time()

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        prediction = my_NMS(prediction)

        if prediction is not None:
            draw_images(batch,load_images,prediction,input_dim)
            print(i)
        else:
            print("No detections were made")





if __name__ == '__main__':
    detect_draw()
