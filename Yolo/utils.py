"'
some image toos 

"'

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

    return anno_dict;

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

def test_img_box():
    rootdir = 'E:\\DataSet\\vehicle_data\\data\\'

    img_rootdir = rootdir + "img"
    img_list = os.listdir(img_rootdir)

    for i in img_list:
        img_path=os.path.join(img_rootdir,i)

        xml_rootdir=rootdir+'annotation'
        xml_name=i+'.xml'
        xml_path=os.path.join(xml_rootdir,xml_name)

        save_rootdir=rootdir+'img_new'
        save_path=os.path.join(save_rootdir,i)

        plot_bbox(img_path,xml_path,save_path)


if __name__=='__main__':
    test_img_box()




