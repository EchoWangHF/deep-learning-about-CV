from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

if __name__=='__main__':
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)