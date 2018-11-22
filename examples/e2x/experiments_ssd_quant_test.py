# -*- coding: utf-8 -*-
"""
Quantitative Comparison of Explanation Models
"""

# standard imports
import numpy as np
import time, os, sys
from scipy.stats import pearsonr

caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
base = caffe_root+'/examples/e2x'
refPda = base + 'pda/results/VOC/VGGNet_300x300/conditional'
refLime = base + 'lime/results/VOC/VGGNet_300x300_slic_1000_10000'
res = dict()

res['e2x_128'] = base + 'intgrads/results_E2X/VOC/VGGNet_300x300_slic_200_128'
res['e2x_64']  = base + 'intgrads/results_E2X/VOC/VGGNet_300x300_slic_200_64'
res['e2x_32']  = base + 'intgrads/results_E2X/VOC/VGGNet_300x300_slic_200_32'
res['e2x_16']  = base + 'intgrads/results_E2X/VOC/VGGNet_300x300_slic_200_16'
res['e2x_8']   = base + 'intgrads/results_E2X/VOC/VGGNet_300x300_slic_200_8'

for key, value in res.iteritems():
    resPath = value
    cLimeAcc = []
    pLimeAcc = []
    cPdaAcc = []
    pPdaAcc = []
    for f in os.listdir(resPath):
        if f.endswith(".npz"):
            #
            refLimeArr = np.load(os.path.join(refLime, f))
            refLimeMask = refLimeArr['arr_0'].reshape(-1)
            refLimeDet = refLimeArr['arr_1']
            refPdaArr = np.load(os.path.join(refPda, f))
            refPdaMask = refPdaArr['arr_0'].reshape(-1)
            refPdaDet = refPdaArr['arr_1']
            estArr = np.load(os.path.join(resPath, f))
            estMask = estArr['arr_0'].reshape(-1)
            estDet = estArr['arr_1']
            assert (estDet[1] - refLimeDet[1]) == 0.0
            assert (estDet[1] - refPdaDet[1]) == 0.0
            #
            cLime, pLime = pearsonr(refLimeMask, estMask)
            cPda, pPda = pearsonr(refPdaMask, estMask)
            if pLime != 1.0:
                cLimeAcc.append(cLime)
                pLimeAcc.append(pLime)
            if pPda != 1.0:
                cPdaAcc.append(cPda)
                pPdaAcc.append(pPda)
    print('LIME/PDA:', key, np.mean(np.array(cLimeAcc)), '/', np.mean(np.array(cPdaAcc)), '|', np.mean(np.array(pLimeAcc)), '/', np.mean(np.array(pPdaAcc)))
