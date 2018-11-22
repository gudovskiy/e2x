# -*- coding: utf-8 -*-
"""
Utility methods for handling data:
"""

import numpy as np
import os
import PIL
import sys

import pandas as pd
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import skimage
from skimage import io

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root)
sys.path.insert(0, caffe_root + 'python')
import caffe
#
def jaccardOverlap(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    N=0 # normalized
    # compute the area of intersection rectangle
    interArea = (xB - xA + N) * (yB - yA + N)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + N) * (boxA[3] - boxA[1] + N)
    boxBArea = (boxB[2] - boxB[0] + N) * (boxB[3] - boxB[1] + N)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    #print interArea, boxAArea, boxBArea
    # return the intersection over union value
    return iou
#
def transformBlob(dataset, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', get_mean(dataset))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return transformer
#
def get_data(dataset, net, folder_name):
    if dataset == "ImageNet":
        #return get_imagenet_data(net)
        print dataset, " is not supported"
    elif dataset == "VOC":
        return get_voc_data(net, folder_name)
    else:
        print dataset, " is not supported"
#
def get_classnames(dataset):
    if dataset == "ImageNet":
        #return np.loadtxt(open('~/data/ilsvrc12/synset_words.txt'), dtype=object, delimiter='\n')
        print dataset, " is not supported"
    elif dataset == "VOC":
        return ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    else:
        print dataset, " is not supported"
#
colors = {  'background' : (0,0,0),

            'aeroplane'      : (39,0,216),
            'bicycle'        : (45,168,97),
            'bird'           : (240,109,0),
            'boat'           : (147,62,197),
            'bottle'         : (88,238,255),

            'bus'            : (0,0,225),
            'car'            : (255,50,50),
            'cat'            : (239,20,216),
            'chair'          : (95,148,57),
            'cow'            : (140,159,0),

            'diningtable'    : (139,10,286),
            'dog'            : (15,222,99),
            'horse'          : (240,109,0),
            'motorbike'      : (147,62,197),
            'person'         : (68,218,255),

            'pottedplant'    : (239,60,116),
            'sheep'          : (35,68,197),
            'sofa'           : (200,10,10),
            'train'          : (247,92,97),
            'tvmonitor'      : (168,118,155)
}
#
def get_color(dataset, classname):
    if dataset == "VOC":
        return colors[classname]
    else:
        print dataset, " is not supported"
#
def get_classnums(dataset):
    """ Returns the number of classes """
    return len(get_classnames(dataset))
#
def get_mean(dataset):
    """ Returns data means """
    return np.array([104, 117, 123])
#
def get_voc_data(net, folder_name):
    """
    Returns a small dataset of ImageNet data.
    Input:
            net         a neural network (caffe model)
    Output:
            blL         network blobs
            gtL         groundtruth
            fnL         filenames with the extension removed
    """
    
    lst = folder_name+'/test_visualization.txt'
    f = open(lst, 'r')
    img_list = f.readlines()
    N = len(img_list)
    data_folder = '/u/big/trainingdata/VOCdevkit/' # points to VOCdevkit
    print "Number of images in dataset = ", N
    # fill up data list
    B,C,H,W = net.blobs['data'].data.shape
    blL = np.zeros((N,C,H,W))
    gtL = []
    fnL = []
    #
    transformer = transformBlob("VOC", net)
    classes = get_classnames("VOC")
    for i in range(N):
        l = img_list[i].strip().split()
        imPath = l[0]
        gtPath = l[1]
        fname = os.path.basename(imPath)
        print i, imPath
        # open image
        im = caffe.io.load_image('{}/{}'.format(data_folder, imPath))
        Y,X,Z = im.shape
        imBlob = transformer.preprocess('data', im)
        # parse gt xml
        gts = []
        xml = ""
        with open('{}/{}'.format(data_folder, gtPath)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        anno = BeautifulSoup(xml)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                cl = str(name_tag.contents[0])
                if cl in classes:
                    bbox = obj.findChildren('bndbox')[0]
                    xmin = int(bbox.findChildren('xmin')[0].contents[0])
                    ymin = int(bbox.findChildren('ymin')[0].contents[0])
                    xmax = int(bbox.findChildren('xmax')[0].contents[0])
                    ymax = int(bbox.findChildren('ymax')[0].contents[0])
                    gts.append([classes.index(cl), xmin, ymin, xmax, ymax])
                    #print cl, xmin, ymin, xmax, ymax
        G = len(gts) # number of groundtruths
        gtBlob = np.zeros((G,8))
        difficult = 0.0 # always not hard
        for g in xrange(0,G):
            gtBlob[g] = [0.0,gts[g][0],0.0,gts[g][1]/float(X),gts[g][2]/float(Y),gts[g][3]/float(X),gts[g][4]/float(Y),difficult]
        # creating lists
        blL[i] = imBlob
        gtL.append(gtBlob)
        fnL.append(fname.replace(".jpg",""))
    #
    return blL, gtL, fnL
##
