import os
import time
import argparse
from utils import *
from darknet import DarkNet, TinyDarkNet
import numpy as np
import gluoncv
import voc_mmAp

image_name = 0


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images',help=
    "Image / Directory containing images to perform detection upon",
                        default="/media/seeking/新加卷/dataset/VL1/val/", type=str)
    parser.add_argument("--video", dest='video', help=
    "video file path", type=str)
    parser.add_argument("--classes", dest="classes", default="my_mxnet_pro_2/data/vcc.names", type=str)
    parser.add_argument("--gpu", dest="gpu", help="gpu id", default="0", type=str)
    parser.add_argument("--dst_dir", dest='dst_dir', help=
    "Image / Directory to store detections to", default="/media/seeking/新加卷/dataset/VL1/save", type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=30, type=int)
    parser.add_argument("--tiny", dest="tiny", help="use yolov3-tiny", default=False, type=bool)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence", default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.5, type=float)
    parser.add_argument("--params", dest='params', help=
    "params file", default="/home/seeking/Vehicle_Data/models/logo_class_2_yolov3_mxnet.params", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)

    return parser.parse_args()


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def draw_bbox(img, bboxs):
    for x in bboxs:
        c1 = tuple(x[1:3].astype("int"))
        c2 = tuple(x[3:5].astype("int"))
        cls = int(x[-1])
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (0, 0, 255)
        label = "{0} {1:.3f}".format(classes[cls], x[-2])
        cv2.rectangle(img, c1, c2, color, 1)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        # c2 = c1[0] + t_size[0] + 2, c1[1] - t_size[1] - 5
        # cv2.rectangle(img, c1, c2, color, -1)
        # cv2.putText(img, label, (c1[0], c1[1] - t_size[1] + 7), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def my_bbox_iou(bbox1,bbox2):

    inter_xmin=nd.max(bbox1[0].asscalar(),bbox2[0].asscalar())
    inter_ymin=nd.max(bbox1[1].asscalar(),bbox2[1].asscalar())

    inter_xmax=nd.min(bbox1[2].asscalar(),bbox2[2].asscalar())
    inter_ymax=nd.min(bbox1[3].asscalar(),bbox2[3].asscalar())

    if(inter_xmax<inter_xmin or inter_ymax<inter_ymin):
        return 0.0
    
    inter_h=inter_ymax-inter_ymin
    inter_w=inter_xmax-inter_xmin
    inter_area=inter_h*inter_w

    bbox1_area=(bbox1[2].asscalar()-bbox1[0].asscalar())*(bbox1[3].asscalar()-bbox1[1].asscalar())
    bbox2_area=(bbox2[2].asscalar()-bbox2[0].asscalar())*(bbox2[3].asscalar()-bbox2[1].asscalar())

    uion_area=bbox1_area+bbox2_area-inter_area

    iou=inter_area/uion_areas
    return iou

def iou_reco(file_path,bboxs):
    if (len(bboxs)!=1):
        return False
    file_path=file_path.split('/')[-1]
    gt_bbox=prase__annotation_xml(file_path)

    pre_bbox=None
    for x in bboxs:
        x=nd.array(x)
        pre_bbox=x[1:5]
    
    pre_bbox=pre_bbox.reshape([1,-1])
    
    # iou_temp=my_bbox_iou(gt_bbox,pre_bbox)
    iou_temp=bbox_iou(gt_bbox,pre_bbox,False)
    if iou_temp>0.5:
        return True
    else:
        return False

def iou_reco2(file_path,bboxs):

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

def cal_iou(imagefile, load_images, output, input_dim):
    
    print('output shape: ',output.shape)

    num_true=0.0

    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])

    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 0], :]

    print('im_dim_list shape: ',im_dim_list.shape)

    scaling_factor = nd.min(input_dim / im_dim_list, axis=1).reshape((-1, 1))
    # scaling_factor = (416 / im_dim_list)[0].view(-1, 1)

    output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim_list[:, 0].reshape((-1, 1))) / 2
    output[:, [2, 4]] -= (input_dim - scaling_factor * im_dim_list[:, 1].reshape((-1, 1))) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i][0].asscalar())
        output[i, [2, 4]] = nd.clip(output[i, [2, 4]], a_min=0.0, a_max=im_dim_list[i][1].asscalar())
    
    num_=output.shape[0]
    for i in range(num_):
        index=int(output[i,0].asscalar())
        res_temp=output[i,1:5]
        if(iou_reco2(imagefile[index],res_temp)):
            num_true+=1
    
    # output=output.asnumpy()
    # for i in range(len(imagefile)):
    #     bboxs=[]
    #     for bbox in output:
    #         if i ==int(bbox[0]):
    #             bboxs.append(bbox)
    #     if(iou_reco(imagefile[i],bboxs)):
    #         num_true+=1
    
    return num_true
    # output = output.asnumpy()
    # for i in range(len(load_images)):
    #     bboxs = []
    #     for bbox in output:
    #         if i == int(bbox[0]):
    #             bboxs.append(bbox)
    #     draw_bbox(load_images[i], bboxs)
    # global image_name
    # list(map(cv2.imwrite, [os.path.join(dst_dir, "{0}.jpg".format(image_name + i)) for i in range(len(load_images))], load_images))
    # image_name += len(load_images)

def save_results(load_images, output, input_dim):

    print('output shape: ',output.shape)

    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])

    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 0], :]

    print('im_dim_list shape: ',im_dim_list.shape)

    scaling_factor = nd.min(input_dim / im_dim_list, axis=1).reshape((-1, 1))
    # scaling_factor = (416 / im_dim_list)[0].view(-1, 1)

    output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim_list[:, 0].reshape((-1, 1))) / 2
    output[:, [2, 4]] -= (input_dim - scaling_factor * im_dim_list[:, 1].reshape((-1, 1))) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = nd.clip(output[i, [1, 3]], a_min=0.0, a_max=im_dim_list[i][0].asscalar())
        output[i, [2, 4]] = nd.clip(output[i, [2, 4]], a_min=0.0, a_max=im_dim_list[i][1].asscalar())

    output = output.asnumpy()
    # assert(len(imagefile)==len(load_images))
    for i in range(len(load_images)):
        bboxs = []
        for bbox in output:
            if i == int(bbox[0]):
                bboxs.append(bbox)
        draw_bbox(load_images[i], bboxs)
    global image_name
    list(map(cv2.imwrite, [os.path.join(dst_dir, "{0}.jpg".format(image_name + i)) for i in range(len(load_images))], load_images))
    image_name += len(load_images)


def predict_video(net, ctx, video_file, anchors):
    if video_file:
        cap = cv2.VideoCapture(video_file)
    else:
        cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    result_video = cv2.VideoWriter(
        os.path.join(dst_dir, "result.avi"),
        cv2.VideoWriter_fourcc("X", "2", "6", "4"),
        25,
        (1280, 720)
    )

    detect_start = time.time()
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_num += 1
            if frame_num % 5 != 0:
                continue
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            img = nd.array(prep_image(frame, input_dim), ctx=ctx).expand_dims(0)

            prediction = predict_transform(net(img), input_dim, anchors)
            prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

            if prediction is None:
                result_video.write(frame)
                continue

            scaling_factor = min(input_dim / frame.shape[0], input_dim / frame.shape[1])

            prediction[:, [1, 3]] -= (input_dim - scaling_factor * frame.shape[1]) / 2
            prediction[:, [2, 4]] -= (input_dim - scaling_factor * frame.shape[0]) / 2
            prediction[:, 1:5] /= scaling_factor

            for i in range(prediction.shape[0]):
                prediction[i, [1, 3]] = nd.clip(prediction[i, [1, 3]], 0.0, frame.shape[1])
                prediction[i, [2, 4]] = nd.clip(prediction[i, [2, 4]], 0.0, frame.shape[0])

            prediction = prediction.asnumpy()
            draw_bbox(frame, prediction)

            result_video.write(frame)

            # cv2.imshow("frame", frame)
            # key = cv2.waitKey(1000)
            # if key & 0xFF == ord('q'):
            #     break
            # print(time.time() - start)
            if frame_num % 100 == 0:
                t = time.time() - detect_start
                print("FPS of the video is {:5.2f}\nPer Image Cost Time {:5.3f}".format(100 / t,
                                                                                        t / 100))
                detect_start = time.time()

        else:
            print("video source closed")
            break
    result_video.release()
    print("{0} detect complete".format(video_file))

def detect_draw():
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim
    dst_dir = args.dst_dir
    start = 0
    classes = load_classes(args.classes)

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)
    
    net = DarkNet(input_dim=input_dim, num_classes=num_classes)

    anchors=my_anchors
    net.initialize(ctx=ctx)
    input_dim = args.input_dim

    try:
        imlist = [os.path.join(images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))

    if not os.path.exists(dst_dir):
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

    # if args.video:
    #     predict_video(net, ctx=ctx, video_file=args.video, anchors=anchors)
    #     exit()

    if not imlist:
        print("no images to detect")
        exit()
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1

    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
                  for i in range(num_batches)]

    output = None
    for i, batch in enumerate(im_batches):
        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)
        start = time.time()

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

        end = time.time()

        if output is None:
            output = prediction
        else:
            output = nd.concat(output, prediction, dim=0)

        print("{0} predicted in {1:6.3f} seconds".format(len(load_images), (end - start) / len(batch)))
        print("----------------------------------------------------------")

        if output is not None:
            save_results(load_images, output, input_dim=input_dim)
        else:
            print("No detections were made")
        output = None

def detect_mAP():
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim
    dst_dir = args.dst_dir

    classes = load_classes(args.classes)

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)

    net = DarkNet(input_dim=input_dim, num_classes=num_classes)

    anchors=my_anchors
    net.initialize(ctx=ctx)
    input_dim = args.input_dim

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

    # output ={}
    eval_mmap=voc_mmAp.VOCMApMetric(0.5)
    for i, batch in enumerate(im_batches):

        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        # prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

        box_corner = nd.zeros(prediction.shape, dtype="float32")
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        prediction_temp=prediction[:,:,:5]

        # cls_name=nd.argmax(prediction[:,:,5:5+num_classes],axis=2)
        cls_name=prediction[:,:,5:5+num_classes]
        confidence_mask=0.5

        T0=cls_name.shape[0]
        T1=cls_name.shape[1]
        T2=cls_name.shape[2]

        for t0 in range(T0):
            for t1 in range(T1):
                for t2 in range(T2):
                    if (cls_name[t0,t1,t2]>confidence_mask):
                        cls_name[t0,t1,t2]=0
                    else:
                        cls_name[t0,t1,t2]=num_classes+1
        
        print(cls_name.shape)

        # cls_name=cls_name.astype("float32").expand_dims(2)
        prediction_out=nd.concat(prediction_temp,cls_name,dim=2)
        print(prediction_out.shape)

        pred_bboxes=prediction_out[:,:,:4]
        pred_scores=prediction_out[:,:,4]
        pred_labels=prediction_out[:,:,5:]

        gt_labels=None
        gt_bboxes=None
        for i,p in enumerate(batch):

            name_temp=batch[i]
            name_temp=name_temp.split('/')[-1]
            gt_temp=prase__annotation_xml(name_temp,classes)
            gt_bboxes_temp=gt_temp[:,:4]
            gt_labels_temp=gt_temp[:,4]

            gt_bboxes_temp=gt_bboxes_temp.astype('float32').expand_dims(0)
            gt_labels_temp=gt_labels_temp.astype('float32').expand_dims(0)

            if gt_bboxes is None:
                gt_bboxes=gt_bboxes_temp
            else:
                gt_bboxes=nd.concat(gt_bboxes,gt_bboxes_temp,dim=0)
            
            if gt_labels is None:
                gt_labels=gt_labels_temp
            else:
                gt_labels=nd.concat(gt_labels,gt_labels_temp,dim=0) 

        eval_mmap.reset()
        eval_mmap.update(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels)
        print(eval_mmap.get())
        
    # np.save('/home/seeking/Vehicle_Data/data/res_file.npy',output)

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
                
    return nd.array(bbox)

def parse_mmAP():
    args = arg_parse()
    res_dict=np.load('/home/seeking/Vehicle_Data/data/res_file.npy').item()

    batch_size=args.batch_size

    classes=load_classes(args.classes)

    gt_labels=None
    gt_bboxes=None
    pred_scores=None
    pred_labels=None
    pred_bboxes=None
    for key in res_dict:

        prediction=res_dict[key]

        pred_bboxes_temp=prediction[:,:4]
        pred_scores_temp=prediction[:,4]
        pred_labels_temp=prediction[:,5]

        pred_bboxes_temp=pred_bboxes_temp.astype('float32').expand_dims(0)
        pred_labels_temp=pred_labels_temp.astype('float32').expand_dims(0)
        pred_scores_temp=pred_scores_temp.astype('float32').expand_dims(0)

        if(pred_bboxes is None):
            pred_bboxes=pred_bboxes_temp
        else:
            pred_bboxes=nd.concat(pred_bboxes,pred_bboxes_temp,dim=0)
        
        if pred_labels is None:
            pred_labels=pred_labels_temp
        else:
            pred_labels=nd.concat(pred_labels,pred_labels_temp,dim=0)
        
        if pred_scores is None:
            pred_scores=pred_scores_temp
        else:
            pred_scores=nd.concat(pred_scores,pred_scores_temp,dim=0)
        
        
        gt_temp=prase__annotation_xml(key,classes)
        gt_bboxes_temp=gt_temp[:,:4]
        gt_labels_temp=gt_temp[:,4]

        gt_bboxes_temp=gt_bboxes_temp.astype('float32').expand_dims(0)
        gt_labels_temp=gt_labels_temp.astype('float32').expand_dims(0)

        if gt_bboxes is None:
            gt_bboxes=gt_bboxes_temp
        else:
            gt_bboxes=nd.concat(gt_bboxes,gt_bboxes_temp,dim=0)
        
        if gt_labels is None:
            gt_labels=gt_labels_temp
        else:
            gt_labels=nd.concat(gt_labels,gt_labels_temp,dim=0) 

    
    # print(gt_bboxes.shape)
    # print(gt_labels.shape)

    # print(pred_bboxes.shape)
    # print(pred_labels.shape)
    # print(pred_scores.shape)

    # eval_mmap=gluoncv.utils.metrics.VOCMApMetric(0.5,classes)
    eval_mmap=voc_mmAp.VOCMApMetric(0.5)

    pred_bboxes=pred_bboxes[:batch_size,:,:]
    pred_labels=pred_labels[:batch_size,:]
    pred_scores=pred_scores[:batch_size,:]
    gt_bboxes=gt_bboxes[:batch_size,:,:]
    gt_labels=gt_labels[:batch_size,:]
    eval_mmap.reset()
    eval_mmap.update(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels)
    print(eval_mmap.get())



def detect_mAP_1():
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim
    dst_dir = args.dst_dir

    classes = load_classes(args.classes)

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)

    net = DarkNet(input_dim=input_dim, num_classes=num_classes)

    anchors=my_anchors
    net.initialize(ctx=ctx)
    input_dim = args.input_dim

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
    iou_confi=0.5
    for i, batch in enumerate(im_batches):

        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

        box_temp=prediction[:,1:5]
        # label_temp=prediction[:,7]

        # if pre_box is None:
        #     pre_box=box_temp
        # else:
        #     pre_box=nd.concat(pre_box,box_temp,dim=0)

        # if pre_label is None:
        #     pre_label=label_temp
        # else:
        #     pre_label=nd.concat(pre_label,label_temp,dim=0)

        gt_bbox=None
        for _,p in enumerate(prediction):
            tag=int(p[0].asscalar())
            # print(tag)
            name_temp=batch[tag]
            name_temp=name_temp.split('/')[-1]
            gt_temp=prase__annotation_xml(name_temp)
            # gt_bboxes_temp=gt_temp[:,:4]
            # gt_labels_temp=gt_temp[:,4]

            # gt_bboxes_temp=gt_bboxes_temp.astype('float32')
            # gt_labels_temp=gt_labels_temp.astype('float32')

            if gt_bbox is None:
                gt_bbox=gt_temp
            else:
                gt_bbox=nd.concat(gt_bbox,gt_temp,dim=0)
            
            # if gt_labels is None:
            #     gt_labels=gt_labels_temp
            # else:
            #     gt_labels=nd.concat(gt_labels,gt_labels_temp,dim=0)
        
        print(box_temp)
        print(gt_bbox)

        iou=bbox_iou(box_temp,gt_bbox,False)


        for i in iou:
            if(i.asscalar()>iou_confi):
                num_true+=1
            num_samples+=1 
    
    # num_samples=float(pre_box.shape[0])
    # num_true=0.0
    # iou_confi=0.5

    # iou=bbox_iou(pre_box,gt_bbox,False)

    # for i in iou:
    #     if(i.asscalar()>iou_confi):
    #         num_true+=1
    
    print(num_true)
    res=num_true/num_samples
    print(res)

def detect_mAP_2():
    # 计算iou 精确度
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    input_dim = args.input_dim
    dst_dir = args.dst_dir
    start = 0
    classes = load_classes(args.classes)

    gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    ctx = try_gpu(args.gpu)[0]
    num_classes = len(classes)
    
    net = DarkNet(input_dim=input_dim, num_classes=num_classes)

    anchors=my_anchors
    net.initialize(ctx=ctx)
    input_dim = args.input_dim

    imlist=[]
    # lablist=[]
    # lab_path='/media/seeking/新加卷/dataset/VL1/annotation'

    for each_i in range(num_classes):
        path_temp=os.path.join(images,classes[each_i])
        image_names=os.listdir(path_temp)
        for image_name_temp in image_names:
            image_=os.path.join(path_temp,image_name_temp.strip())
            
            if (os.path.isfile(image_)==False):
                continue
            else:
                imlist.append(image_)
            
            # label_list_temp=[]
            # label_=os.path.join(lab_path,image_name_temp.strip()+'.xml')
            # if (os.path.isfile(label_)==False):
            #     continue
            # else:
            #     label_list_temp.append(label_)
            #     label_list_temp.append(each_i)
            #     lablist.append(label_list_temp)

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

    # if args.video:
    #     predict_video(net, ctx=ctx, video_file=args.video, anchors=anchors)
    #     exit()

    if not imlist:
        print("no images to detect")
        exit()
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1

    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
                  for i in range(num_batches)]

    output = None
    num_true=0.0
    num_samples=0.0
    for i, batch in enumerate(im_batches):
        load_images = [cv2.imread(img) for img in batch]
        tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
        tmp_batch = nd.array(tmp_batch, ctx=ctx)
        start = time.time()

        net_result=net(tmp_batch)

        prediction = predict_transform(net_result, input_dim, anchors)

        prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

        if output is None:
            output = prediction
        else:
            output = nd.concat(output, prediction, dim=0)
        
        # print("{0} predicted in {1:6.3f} seconds".format(len(load_images), (end - start) / len(batch)))
        # print("----------------------------------------------------------")
        num_samples=num_samples+len(batch)
        if output is not None:
            num_true_temp=cal_iou(batch,load_images, output, input_dim=input_dim)
            num_true=num_true+num_true_temp
        else:
            print("No detections were made")
        output = None
    
    print(num_true)
    print(num_true/num_samples)



if __name__ == '__main__':
    detect_mAP_2()

    # args = arg_parse()
    # images = args.images
    # batch_size = args.batch_size
    # confidence = args.confidence
    # nms_thresh = args.nms_thresh
    # input_dim = args.input_dim
    # dst_dir = args.dst_dir
    # start = 0
    # classes = load_classes(args.classes)

    # gpu = [int(x) for x in args.gpu.replace(" ", "").split(",")]
    # ctx = try_gpu(args.gpu)[0]
    # num_classes = len(classes)
    
    # net = DarkNet(input_dim=input_dim, num_classes=num_classes)

    # anchors=my_anchors
    # net.initialize(ctx=ctx)
    # input_dim = args.input_dim

    # imlist=[]
    # for each_i in range(num_classes):
    #     path_temp=os.path.join(images,classes[each_i])
    #     image_names=os.listdir(path_temp)
    #     for image_name_temp in image_names:
    #         image_=os.path.join(path_temp,image_name_temp.strip())
            
    #         if (os.path.isfile(image_)==False):
    #             continue
    #         else:
    #             imlist.append(image_)

    # if not os.path.exists(dst_dir):
    #     os.mkdir(dst_dir)

    # if args.params.endswith(".params"):
    #     net.load_parameters(args.params)
    # elif args.params.endswith(".weights"):
    #     tmp_batch = nd.uniform(shape=(1, 3, args.input_dim, args.input_dim), ctx=ctx)
    #     net(tmp_batch)
    #     net.load_weights(args.params, fine_tune=False)
    # else:
    #     print("params {} load error!".format(args.params))
    #     exit()
    # print("load params: {}".format(args.params))
    # net.hybridize()

    # # if args.video:
    # #     predict_video(net, ctx=ctx, video_file=args.video, anchors=anchors)
    # #     exit()

    # if not imlist:
    #     print("no images to detect")
    #     exit()
    # leftover = 0
    # if len(imlist) % batch_size:
    #     leftover = 1

    # num_batches = len(imlist) // batch_size + leftover
    # im_batches = [imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]
    #               for i in range(num_batches)]

    # output = None
    # for i, batch in enumerate(im_batches):
    #     load_images = [cv2.imread(img) for img in batch]
    #     tmp_batch = list(map(prep_image, load_images, [input_dim for x in range(len(batch))]))
    #     tmp_batch = nd.array(tmp_batch, ctx=ctx)
    #     start = time.time()

    #     net_result=net(tmp_batch)

    #     prediction = predict_transform(net_result, input_dim, anchors)

    #     prediction = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

    #     end = time.time()

    #     if output is None:
    #         output = prediction
    #     else:
    #         output = nd.concat(output, prediction, dim=0)

    #     print("{0} predicted in {1:6.3f} seconds".format(len(load_images), (end - start) / len(batch)))
    #     print("----------------------------------------------------------")

    #     if output is not None:
    #         save_results(load_images, output, input_dim=input_dim)
    #     else:
    #         print("No detections were made")
    #     output = None
