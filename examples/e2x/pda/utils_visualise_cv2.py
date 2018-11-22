# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation
"""

from skimage import color
import numpy as np
import cv2
import utils_detectors as utlC
import utils_data as utlD

def plot_ssd(method, im, dataset, mask, l, det, save_path):
    '''
    Plot the results of the relevance estimation
    '''
    H,W,C = im.shape
    classes = utlD.get_classnames(dataset)
    classNum = utlD.get_classnums(dataset) # num of classes
    trueClassName = classes[int(det[1])]
    lName = classes[l]
    result = utlC.decode_ssd_result(det[3:5])
    score = str(round(100*det[2]))
    [xmin, ymin, xmax, ymax] = np.round( det[5:9] * np.array([W, H, W, H]) ).astype(int)
    colorsB = np.linspace(250, 100, num=classNum).astype(int)
    colorsG = np.linspace(100, 250, num=classNum).astype(int)
    colorsR = np.linspace(0  , 255, num=classNum).astype(int)
    # original image
    dIm = im.copy()
    cv2.rectangle(dIm, (xmin, ymin), (xmax, ymax), (0,255,255), 2)#, (colorsB[l],colorsG[l],colorsR[l]), 1)
    # heatmap
    p = mask
    pNorm = np.max(np.abs(p))+1e-8 # avoid dividing by zero
    pIm = 255 * np.ones((im.shape))
    #
    pB = 255.0*p.copy()/pNorm
    pR = 255.0*p.copy()/pNorm
    pB[pB>0.0] = 0.0
    pR[pR<0.0] = 0.0

    pIm[:,:,0] -= pR
    pIm[:,:,1] -= pR
    pIm[:,:,2] -= pR/4
    
    pIm[:,:,0] += pB/4
    pIm[:,:,1] += pB
    pIm[:,:,2] += pB
    # bar image
    bIm = np.zeros((H,20,C))
    bIm[0:H/2,::] = np.stack( (np.tile(np.linspace(0, 255, num=H/2)[..., np.newaxis],20), np.tile(np.linspace(0, 255, num=H/2)[..., np.newaxis],20), np.tile(np.linspace(192, 255, num=H/2)[..., np.newaxis],20)), axis=2 )
    bIm[H/2:H,::] = np.stack( (np.tile(np.linspace(255, 192, num=H/2)[..., np.newaxis],20), np.tile(np.linspace(255, 0, num=H/2)[..., np.newaxis],20), np.tile(np.linspace(255, 0, num=H/2)[..., np.newaxis],20)), axis=2 )
    # overlayed image
    gIm = 0.299*im[:,:,2]+0.587*im[:,:,1]+0.114*im[:,:,0]
    oIm = np.tile(gIm[..., np.newaxis],3)
    alpha = 0.6
    cv2.addWeighted(pIm, alpha, im, 1-alpha, 0, oIm)
    cv2.rectangle(oIm, (xmin, ymin), (xmax, ymax), (0,255,255), 2)#, (colorsB[l],colorsG[l],colorsR[l]), 1)
    # draw
    back = np.tile(200, 3) # color of the border
    im1 = cv2.copyMakeBorder(dIm, 40,10,10,10, cv2.BORDER_CONSTANT, value=back)
    im2 = cv2.copyMakeBorder(pIm, 40,10,10,10, cv2.BORDER_CONSTANT, value=back)
    im3 = cv2.copyMakeBorder(bIm, 40,10,10,90, cv2.BORDER_CONSTANT, value=back)
    im4 = cv2.copyMakeBorder(oIm, 40,10,10,10, cv2.BORDER_CONSTANT, value=back)
    #
    text1 = trueClassName + '(' + result + ', conf=' + score + '%)'
    text2 = method + ': ' + lName
    text3 = ' '
    textP3 = "+{:1.6f}".format(pNorm)
    textZ3 = " {:1.1f}".format(0.0)
    textN3 = "-{:1.6f}".format(pNorm)
    text4 = 'Overlayed'
    #
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    textOrg = (12,22)
    #
    cv2.putText(im1, text1, textOrg,    fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im2, text2, textOrg,    fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, text3, textOrg,    fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textP3,(30,0+40),  fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textZ3,(30,H/2+40),fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textN3,(30,H+40),  fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im4, text4, textOrg,    fontFace, fontScale, (0,0,0), thickness, 0, bottomLeftOrigin=False)
    #
    finalIm = cv2.hconcat((im1, im2, im3, im4))
    cv2.imwrite(save_path + '.png', finalIm)
