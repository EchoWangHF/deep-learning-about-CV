import os
import xml.etree.ElementTree as ET
import cv2

'''
<?xml version="1.0" encoding="UTF-8"?>
-<Anontations>
-<车>
     <x>293</x>
     <y>93</y>
     <w>272</w>
     <h>227</h>
</车>
-<车标>
     <x>417</x>
     <y>244</y>
     <w>29</w>
     <h>28</h>
</车标>
-<车牌>
    <x>403</x>
    <y>278</y>
    <w>63</w>
    <h>21</h>
</车牌>
</Anontations>
'''
def read_xml(xml_path):
    if(os.path.isfile(xml_path)==False):
        print('xml is not exists')
        return
    tree=ET.parse(xml_path)

    root=tree.getroot()

    anno_dict={} #save the bbox;

    for i in root:
        box_list=[]

        for each in i:
            box_list.append(int(each.text))

        anno_dict[i.tag]=box_list

    return anno_dict

def plot_bbox(img_path,xml_path,save_path):
    dict=read_xml(xml_path)
    if dict is None:
        print('dict is null')
        return

    img=cv2.imread(img_path)
    if img is None:
        print('image is null')
        return

    red=[0,0,255]
    for i in dict:
        name=i
        box=dict[name]
        x=box[0]
        y=box[1]
        w=box[2]
        h=box[3]

        cv2.rectangle(img,(x,y),(x+w,y+h),red,1)

    cv2.imwrite(save_path,img)
    return

def transfrom_bbox(dict):
    dict['车标'][0]-=dict['车'][0]
    dict['车标'][1]-=dict['车'][1]

    dict['车牌'][0]-=dict['车'][0]
    dict['车牌'][1]-=dict['车'][1]

    # box_car=dict['车']
    # box_logo=dict['车标']
    # box_plante=dict['车牌']

    # box_logo[0]-+box_car[0]
    # box_logo[1]-+box_car[1]

    # box_plante[0]-+box_car[0]
    # box_plante[1]-+box_car[1]
    del dict['车']
    return dict

def cut_img(img_path,xml_path,save_path):
    dict=read_xml(xml_path)
    if dict is None:
        print('dict is null')
        return
    
    img=cv2.imread(img_path)
    if img is None:
        print('image is null')
        return
    
    
    for i in dict:
        name=str(i)
        if(name=='车'):
            box=dict[name]
            x=box[0]
            y=box[1]
            w=box[2]
            h=box[3]

            img=img[y:y+h,x:x+w]

            cv2.imwrite(save_path,img)
            break
        else:
            continue

def test_img_box():
    rootdir = '/home/seeking/Vehicle_Data/data'

    img_rootdir = os.path.join(rootdir,'val')
    img_list = os.listdir(img_rootdir)

    for i in img_list:
        img_path=os.path.join(img_rootdir,i)

        xml_rootdir=os.path.join(rootdir,'annotation')
        xml_name=i+'.xml'
        xml_path=os.path.join(xml_rootdir,xml_name)

        save_rootdir=os.path.join(rootdir,'val_car')
        save_path=os.path.join(save_rootdir,i)

        cut_img(img_path,xml_path,save_path)


if __name__=='__main__':
    test_img_box()