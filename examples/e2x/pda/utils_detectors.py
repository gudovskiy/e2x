# -*- coding: utf-8 -*-
"""

Utility methods for handling the detectors:
    set_caffe_mode(gpu)
    get_caffenet(netname)
    forward_pass(net, x, blobnames='prob', start='data')

"""

# this is to supress some unnecessary output of caffe in the linux console
import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import sys
caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root)
sys.path.insert(0, caffe_root + 'python')
import caffe
#
def set_caffe_mode(gpu_id):
    ''' Set whether caffe runs in gpu or not, input is boolean '''
    if gpu_id == None:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
#
def get_caffenet(dataset, netname):

    p = os.environ["CAFFE_ROOT"] + '/models'
    if dataset == "ImageNet":
        print dataset, ' model is not found'
    elif dataset == "VOC":
        blobnames = ['mbox_conf_flatten'] # SSD output layer
        if netname=='VGGNet_300x300':
            model_path = p + '/VGGNet/VOC0712/SSD_300x300/'
            param_fn = model_path + 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
        elif netname=='VGGNet_512x512':
            model_path = p + '/VGGNet/VOC0712/SSD_512x512/'
            param_fn = model_path + 'VGG_VOC0712_SSD_512x512_iter_130000.caffemodel'
        else:
            print dataset, ' model is not found'
    else:
        print dataset, " is not supported"
    #
    netP_fn = model_path + 'deploy_viz.prototxt' # prediction model network
    netP = caffe.Net(netP_fn,  param_fn, caffe.TEST)
    netD_fn = model_path + 'deploy_pda.prototxt' # explanation model network
    netD = caffe.Net(netD_fn,  param_fn, caffe.TEST)
    #
    return netP, netD, blobnames
#
def get_demonet(dataset, netname):
    param_fn = os.environ["CAFFE_ROOT"] + '/models/VOC0712Plus/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
    netD_fn = os.environ["CAFFE_ROOT"] + '/models/VOC0712Plus/SSD_300x300/deploy_pda.prototxt' # explanation model network
    netD = caffe.Net(netD_fn, param_fn, caffe.TEST)
    blobnames = ['detection_out', 'mbox_conf_flatten'] # SSD output layer
    return netD, blobnames
#
def decode_ssd_result(d):
    '''
    
    '''
    if all(d == np.array([1,0])):
        result = 'TP'
    elif all(d == np.array([0,1])):
        result = 'FP'
    elif all(d == np.array([1,1])):
        result = 'FN'
    else:
        print 'Something is wrong in detection!'

    return result

#
def forward_pass(net, x, blobnames='prob', start='data'):
    '''
    Defines a forward pass (modified for our needs)
    Input:      net         the network (caffe model)
                x           the input, a batch of imagenet images
                blobnames   for which layers we want to return the output,
                            default is output layer ('prob')
                start       in which layer to start the forward pass
    '''

    # get input into right shape
    if np.ndim(x)==3:
        x = x[np.newaxis]
    if np.ndim(x)<4:
        input_shape = net.blobs[start].data.shape
        x = x.reshape([x.shape[0]]+list(input_shape)[1:])

    # reshape net so it fits the batchsize (implicitly given by x)
    if net.blobs['data'].data.shape[0] != x.shape[0]:
        net.blobs['data'].reshape(*(x.shape))
    # feed forward the batch through the next
    net.forward_all(data=x)
    # collect outputs of the blobs we're interested in
    returnVals = [np.copy(net.blobs[b].data[:]) for b in blobnames]

    return returnVals
#
def backward_pass(net, index, blobnames='prob', start='data'):
    '''
    Defines a backward pass (modified for our needs)
    Input:      net         the network (caffe model)
                index       the input index
                blobnames   for which layers we want to return the output,
                            default is output layer ('prob')
                start       in which layer to start the forward pass
    '''
    
    for b in blobnames:
        net.blobs[b].diff[:]        = np.zeros(net.blobs[b].diff.shape)
        net.blobs[b].diff[:, index] = np.ones( net.blobs[b].diff.shape[0])

    gradApprox = net.backward()[start]

    return gradApprox
