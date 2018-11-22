# -*- coding: utf-8 -*-
"""
PDA for SSD
"""

# standard imports
import numpy as np
import time, os, sys, datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# utilities
from prediction_difference_analysis_ssd import PredDiffAnalyser
#
method = 'PDA'
confidence_threshold = 0.5 # confidence threshold should be like in your real detector. threshold in deploy_viz.prototxt should be low to cover all detections including FNs.
#
def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description='PDA for SSD')

    parser.add_argument('--folder_name', dest='folder_name', help='Folder to analyze. Example: python ./examples/e2x/pda/experiments_ssd.py --folder_name ./data/VOC0712',
                        default='~/gitlab/ssd/data/VOC0712', type = str)

    parser.add_argument('--folder_est_name', dest='folder_est_name', help='folder that points to samples used for estimation of conditional sampling parameters', type=str)
    
    parser.add_argument('--sampl_style', dest='sampl_style', help='choose auto/conditional/marginal sampling',
                        default='conditional', type=str)

    parser.add_argument('--dataset', dest='dataset', help='choose dataset which was used to train network: ImageNet/VOC/KITTI',
                        default='VOC', type=str)

    parser.add_argument('--net_name', dest='net_name', help='choose network to analyze',
                        default='VGGNet_300x300', type=str)
                        # VGGNet_512x512

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use',
                        default=0, type=int)

    parser.add_argument('--batch_size', dest='batch_size', help='if caffe crashes with memory error, reduce the batch size',
                        default=16, type=int)

    parser.add_argument('--win_size', dest='win_size', help='window size k in alg 1 (see paper), 0 means automatic selection',
                        default=0, type=int)

    parser.add_argument('--overlapping', dest='overlapping', help='1/0 indicate whether windows should be overlapping or not',
                        default=True, type=bool)
    
    parser.add_argument('--stride', dest='stride', help='strided analysis to analyze only one pixel for each stride value, 0 means automatic selection',
                        default=0, type=int)

    parser.add_argument('--num_samples', dest='num_samples', help='number of samples for conditional sampling (always 1 for marginal)',
                        default=10, type=int)

    parser.add_argument('--padding_size', dest='padding_size', help='for conditional sampling: l = win_size+2*padding_size in alg 1',
                        default=1, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # make sure there is this environment variable
    caffe_root = os.environ["CAFFE_ROOT"]
    os.chdir(caffe_root)
    print caffe_root
    sys.path.append(caffe_root+'/examples/e2x/pda')
    import utils_detectors as utlC
    import utils_data as utlD
    import utils_sampling as utlS
    import utils_visualise_cv2 as utlV
    #
    args = parse_args()
    folder_name = args.folder_name
    if args.folder_est_name is None:
        folder_est_name = args.folder_name
    else:
        folder_est_name = args.folder_est_name
    sampl_style = args.sampl_style
    dataset = args.dataset
    net_name = args.net_name
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    win_size = args.win_size
    overlapping = args.overlapping
    stride = args.stride
    num_samples = args.num_samples
    padding_size = args.padding_size
    #
    utlC.set_caffe_mode(gpu_id)
    netP, netD, blobnames = utlC.get_caffenet(dataset, net_name)
    # get the data
    blL, gtL, fnL = utlD.get_data(dataset, netP, folder_name)
    #
    test_indices = range(len(fnL))
    # get the label names
    classes = utlD.get_classnames(dataset)
    CC = utlD.get_classnums(dataset) # num of classes
    C = CC-1 # num of classes - background
    # make folder for saving the results if it doesn't exist
    path_results = './examples/e2x/pda/results/{}/{}_{}'.format(dataset, net_name, sampl_style)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    # ------------------------ EXPERIMENTS ------------------------
    # target function (mapping input features to output confidences)
    [TB, TC, TH, TW] = netD.blobs['data'].data.shape
    mean = utlD.get_mean(dataset)
    netD.blobs['data'].reshape(batch_size, TC, TH, TW)
    target_forward_func = lambda x: utlC.forward_pass(netD, x, blobnames)
    # for the given test indices, do the prediction difference analysis
    print('START TIME', datetime.datetime.time(datetime.datetime.now()))
    run_count = 0
    for i in test_indices:
        # get net input
        bl = np.squeeze(blL[i])
        netP.blobs['data'].data[:] = bl
        # back from blob to image
        im = np.copy(bl)
        im += np.tile(mean[:, np.newaxis, np.newaxis],(1, im.shape[1], im.shape[2]))
        im = im.transpose((1,2,0)) # CHW->HWC
        # get net groundtruth
        G = gtL[i].shape[0] # number of groundtruths
        netP.blobs['label'].reshape(1,1,G,8)
        netP.blobs['label'].data[0,0] = gtL[i]
        fwP = netP.forward()
        dets = fwP['detection_eval'][0,0]
        dets = dets[C:,:] # skip class declaration
        B = dets.shape[0] # number of detected bboxes
        # process detections
        gtDets = gtL[i]
        TP = 0 # true positives
        FN = 0 # false negatives
        FP = 0 # false positives
        listTP = []
        listFN = []
        listFP = []
        for b in xrange(0,B):
            if (utlC.decode_ssd_result(dets[b,3:5]) == 'TP'):
                if (dets[b,2] > confidence_threshold):
                    TP += 1
                    listTP.append(b) # index of TPs
                else:
                    FN += 1
                    dets[b,4] = 1.0 # make it FN
                    listFN.append(b) # index of FNs
            elif (utlC.decode_ssd_result(dets[b,3:5]) == 'FP') and (dets[b,2] > confidence_threshold):
                FP += 1
                listFP.append(b) # index of FPs
        # sanity check
        if (TP+FN) != G:
            print 'Some FNs are not covered due to low probability: (TP+FN) =', TP+FN, 'should be equal to G =', G
        # construct filtered list of detections
        BB = TP+FP+FN
        detsBB = np.zeros((BB, dets.shape[1]))
        detsBB[0:TP    ] = dets[listTP]
        detsBB[TP:TP+FP] = dets[listFP]
        detsBB[TP+FP:BB] = dets[listFN]
        # print statistics
        print "Statistics for", fnL[i], "Number of GTs = ", G,', TP = ', TP, ', FP = ', FP, ', FN = ', FN
        # do analysis for EACH detection
        for b in xrange(BB):
            #
            run_count = run_count + 1
            #
            label = detsBB[b, 1]
            labelName = classes[int(label)]
            pconf = detsBB[b, 2]
            result = utlC.decode_ssd_result(detsBB[b, 3:5])
            # automatic crop selection based on bbox
            xmin,ymin,xmax,ymax = detsBB[b,5:9]
            xd = xmax-xmin
            yd = ymax-ymin
            boxArea = xd*yd
            #
            det = detsBB[b]
            index = int(detsBB[b, 9])
            # automatic window/stride selection based on bbox
            winSize = win_size
            if win_size == 0:
                baseWinSize = np.sqrt(TH*TW)/6.0 # 1/6th
                winSize = max( 1, int( np.round(baseWinSize * np.sqrt(boxArea) ) ) ) # always >= 1
                #print 'BOX/WIN SIZE: ',boxArea,winSize
            strideAdapt = stride
            if stride == 0:
                baseStride = np.sqrt(TH*TW)/6.0 # 1/6th
                strideAdapt = max( 1, int( np.round(baseStride * np.sqrt(boxArea) ) ) ) # always >= 1
                #print 'BOX/STRIDE SIZE: ',boxArea,strideAdapt
            #
            samplStyle = sampl_style
            if samplStyle == 'conditional':
                sampler = utlS.cond_sampler(winSize, padding_size, [TH, TW], dataset, netD, net_name, folder_name, folder_est_name)
                numSamples = num_samples
                if batch_size < numSamples:
                    sys.exit('Error: batch_size should be more or equal to numSamples!')
            elif samplStyle == 'marginal':
                sampler = utlS.marg_sampler(bl)
                numSamples = 1
            #
            pda = PredDiffAnalyser(bl, target_forward_func, sampler, CC, index, num_samples=numSamples, batch_size=batch_size)
            print "analyzing detection {} out of {} ({}) for file: {}, class: {}, confidence: {}, window size: {}, stride: {}, sampling: {}" \
                .format(b,BB,result,fnL[i],labelName,pconf,winSize,strideAdapt,samplStyle)
            start_time = time.time()
            pred_diff = pda.get_rel_vect(gt=det, gt_dims=[TH, TW], win_size=winSize, overlap=overlapping, stride=strideAdapt)
            # get the path for saving the results
            if 1:
                #save_path = path_results+'/{}_{}'.format(fnL[i], b)
                save_path = path_results+'/{}_{}_winSize{}_stride{}'.format(fnL[i], b, winSize, strideAdapt)
                if os.path.exists(save_path+'.png'):
                    print 'Results for', save_path, ' exist, will move to the next detection.'
                    continue
                # plot and save the results
                l = int(label)
                maskSeg = pred_diff[0][:,l].reshape((TH,TW))
                utlV.plot_ssd(method, im, dataset, maskSeg, l, det, save_path)
                np.savez(save_path, maskSeg, det)
            print "--- Total time took {:.3f} seconds ---".format((time.time() - start_time))
    #
    print('END TIME', datetime.datetime.time(datetime.datetime.now()), run_count)
    del netP, netD, blL, fnL
