import os
import time
import argparse
from utils import *
from darknet4 import DarkNet
import gluoncv

image_name = 0


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images',help=
    "Image / Directory containing images to perform detection upon",
                        default="/home/seeking/Vehicle_Data/data/val/", type=str)
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
    "params file", default="my_mxnet_pro_3/data/pro_darknet4_mxnet.params", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)

    return parser.parse_args()

def iou_reco(file_path,bboxs):

    file_path=file_path.split('/')[-1]
    gt_bbox=prase__annotation_xml(file_path)

    pre_bbox=bboxs[:4]
    
    pre_bbox=pre_bbox.reshape([1,-1])
    
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

def my_NMS(prediction):
    batch_size = prediction.shape[0]

    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output=None

    for ind in range(batch_size):
        image_pred=prediction[ind]
        rows=image_pred.shape[0]
        max_index=nd.argmax(image_pred[:,4],axis=0)
        max_index=int(max_index.asscalar())
        if(max_index>=rows):
            print(max_index,ind)
            continue
        else:
            image_temp=image_pred[max_index,:5]
        image_temp[4]=ind
        image_temp=image_temp.expand_dims(0)
        if output is None:
            output=image_temp
        else:
            output=nd.concat(output,image_temp,dim=0)
    
    return output


def detect_mAP():
    # 计算iou 精确度
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim

    classes = load_classes(args.classes)

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)
    
    net = DarkNet(input_dim=input_dim, num_classes=num_classes)

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
        try:
            tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        except:
            print(batch)
            return

        tmp_batch = nd.array(tmp_batch, ctx=ctx)

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        if(prediction.shape[0]!=len(batch)):
            continue

        prediction = my_NMS(prediction)
    
        num_samples=num_samples+len(batch)
        if prediction is not None:
            num_true_temp=cal_iou(batch,load_images,prediction, input_dim=input_dim)
            num_true=num_true+num_true_temp
        else:
            print("No detections were made")
    
    print(num_true)
    print(num_true/num_samples)



if __name__ == '__main__':
    detect_mAP()