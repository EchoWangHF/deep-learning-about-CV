import os 
import numpy as np
import xml.etree.ElementTree as ET
import cv2

def Train_File():
    file_path='F:\\ILSVRC2017_DET\\ILSVRC\\Annotations\\DET\\train\\ILSVRC2013_train\\'
    index=0
    for i in os.listdir(file_path):

        num=0
        for j in os.listdir(file_path+str(i)):
            xml_path=file_path+str(i)+'\\'+str(j)
            xml_list=Read_train_Xml(xml_path,index)
            if(xml_list==[]):
                continue
            bndBox_Image(xml_list,num)
            xml_list.clear()
            num+=1
        print(index)
        index+=1

def Val_File():
    train_file_path='F:\\ILSVRC2017_DET\\ILSVRC\\Annotations\\DET\\train\\ILSVRC2013_train\\'
    index=0
    Train_dict={}
    for i in os.listdir(train_file_path):
        Train_dict[i]=index
        index+=1
    
    val_file_path='F:\\ILSVRC2017_DET\\ILSVRC\\Annotations\\DET\\val\\'
    num=0
    for i in os.listdir(val_file_path):
        xml_path=val_file_path+i
        xml_list=Read_val_Xml(xml_path,-1)
        if(xml_list==[]):
            continue
        
        xml_list[6]=Train_dict[xml_list[1]]
        # print(xml_list)

        bndBox_Image(xml_list,num)
        xml_list.clear()
        num+=1

    
def Read_val_Xml(val_xml,index):
    tree = ET.parse(val_xml)
    root=tree.getroot()
    
    xml_list=[]

    try:
        filename=root.findall('./filename')
        xml_list.append(filename[0].text)

        name=root.findall('./object/name')
        xml_list.append(name[0].text)

        xmin=root.findall('./object/bndbox/xmin')
        xml_list.append(xmin[0].text)

        xmax=root.findall('./object/bndbox/xmax')
        xml_list.append(xmax[0].text)

        ymin=root.findall('./object/bndbox/ymin')
        xml_list.append(ymin[0].text)

        ymax=root.findall('./object/bndbox/ymax')
        xml_list.append(ymax[0].text)

        xml_list.append(index)
    except:
        print(val_xml)
        xml_list=[]
    return xml_list



    

def Read_train_Xml(xml_path,index):
    tree = ET.parse(xml_path)
    root=tree.getroot()
    
    xml_list=[]

    try:
        folder=root.findall('./folder')
        xml_list.append(folder[0].text)

        filename=root.findall('./filename')
        xml_list.append(filename[0].text)

        xmin=root.findall('./object/bndbox/xmin')
        xml_list.append(xmin[0].text)

        xmax=root.findall('./object/bndbox/xmax')
        xml_list.append(xmax[0].text)

        ymin=root.findall('./object/bndbox/ymin')
        xml_list.append(ymin[0].text)

        ymax=root.findall('./object/bndbox/ymax')
        xml_list.append(ymax[0].text)

        xml_list.append(index)
    except:
        print(xml_path)
        xml_list=[]
    return xml_list

def bndBox_Image(xml_list,num):
    # image_path='F:\\ILSVRC2017_DET\\ILSVRC\Data\\DET\\train\\ILSVRC2013_train\\'+xml_list[0]+'\\'+xml_list[1]+'.JPEG'
    image_path='F:\\ILSVRC2017_DET\\ILSVRC\Data\\DET\\val\\'+xml_list[0]+'.JPEG'
    img=cv2.imread(image_path,cv2.COLOR_BGR2RGB)
    if(img==[]):
        print(xml_list)
        return
    img2=img[int(xml_list[4]):int(xml_list[5]),int(xml_list[2]):int(xml_list[3])]

    img_write_path='F:\\ImageNet\\val\\'+str(xml_list[6])+"n"+str(num)+'.JPEG'
    cv2.imwrite(img_write_path,img2)


if __name__=='__main__':
    # Val_File()
    Train_File()
    # xml=Read_Xml('F:\\ILSVRC2017_DET\\ILSVRC\\Annotations\\DET\\train\\ILSVRC2013_train\\n02107908\\n02107908_9.xml',0)
    # bndBox_Image(xml,1)
