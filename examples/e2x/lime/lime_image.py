"""
Functions for explaining classifiers that use Image data.
"""
import copy

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import lime_base
from wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: TODO

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            mask = np.zeros(segments.shape, segments.dtype)
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            mask = np.zeros(segments.shape)
            wSum = 0.0
            for f,w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                #c = 0 if w < 0 else 1
                mask[segments == f] = w # 1 if w < 0 else 2
                wSum += w
                #mask[segments == f] = w/(np.sum([segments == f])) # 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                #temp[segments == f, c] = np.max(image)
                #for cp in [0, 1, 2]:
                #    if c == cp:
                #        continue
                #    # temp[segments == f, cp] *= 0.5
            #print('EXP = ', len(exp), wSum)
        #else:
        #    for f, w in exp[:num_features]:
        #        if np.abs(w) < min_weight:
        #            continue
        #        c = 0 if w < 0 else 1
        #        mask[segments == f] = 1 if w < 0 else 2
        #        temp[segments == f] = image[segments == f].copy()
        #        temp[segments == f, c] = np.max(image)
        #        for cp in [0, 1, 2]:
        #            if c == cp:
        #                continue
        #            # temp[segments == f, cp] *= 0.5
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)

    def explain_instance(self, image, blob, det, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=16,
                         segmentation_fn=None, n_segments=200,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        #
        seg_fn = None
        if segmentation_fn == 'quickshift':
            seg_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2, random_seed=random_seed)#, sigma=1)
        elif segmentation_fn == 'felzenszwalb':
            seg_fn = SegmentationAlgorithm('felzenszwalb')
        elif segmentation_fn == 'slic':
            seg_fn = SegmentationAlgorithm('slic', n_segments=n_segments)#, compactness=10)#, sigma=1)
        elif segmentation_fn == 'uniform':
            #seg_fn = SegmentationAlgorithm('uniform', n_segments=n_segments)
            print(segmentation_fn+' is a custom segmentation fn')
        else:
            print('segmentation_fn is wrong!')
        #
        if segmentation_fn == 'uniform':
            H,W,_ = image.shape
            p = np.floor(np.sqrt(H*W/n_segments))
            segments = np.zeros((H,W), dtype=int)
            WW = np.floor(W/p)
            for h in range(H):
                for w in range(W):
                    segment = np.floor(w/p) + WW*np.floor(h/p)
                    segments[h,w] = segment.astype(int)
                #print(segments[h,:])
        else:
            try:
                segments = seg_fn(image)
            except ValueError as e:
                raise e

        '''H,W,_ = image.shape
        [xmin, ymin, xmax, ymax] = np.round( det[5:9] * np.array([W,H,W,H]) ).astype(int)
        mask = -np.ones((H,W)).astype(int)
        d = np.round(np.array([xmax-xmin, ymax-ymin]) / 2)
        c = np.array([xmin, ymin]) + d
        [x0,y0] = c - 3*d
        [x1,y1] = c + 3*d
        x0 = max(0,min(W, x0))
        x1 = max(0,min(W, x1))
        y0 = max(0,min(H, y0))
        y1 = max(0,min(H, y1))
        mask[y0:y1,x0:x1] = 1
        segments *= mask'''
        #
        #print(segments.shape, segments[0,:])
        #plt.figure() plt.imshow(image)
        #plt.figure() plt.imshow(mark_boundaries(image, segments))
        #plt.show()
        #
        fudged_blob = blob.copy()
        if hide_color is None:
            for x in np.unique(segments):
                #print(np.mean(blob[0][segments == x]), np.mean(blob[1][segments == x]), np.mean(blob[2][segments == x]))
                fudged_blob[0][segments == x] = np.mean(blob[0][segments == x])
                fudged_blob[1][segments == x] = np.mean(blob[1][segments == x])
                fudged_blob[2][segments == x] = np.mean(blob[2][segments == x])
        else:
            fudged_blob[:] = hide_color

        top = labels

        data, labels = self.data_labels(blob, fudged_blob, det, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)
        print('Data shape =', data.shape)
        labels = labels[:, top]
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    det,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        #print(n_features)
        #print(np.unique(segments).shape[0], np.unique(segments[segments>0]).shape[0], segments.shape, segments[np.nonzero(segments)].shape)
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features)) # sample from uniform binomial distribution
        #prob = np.ones(n_features)/10.0 # probability of each feature
        #data = np.zeros((num_samples, n_features)).astype(int)
        #for n in range(n_features):
        #    data[:,n] = self.random_state.choice(2, num_samples, p=[1-prob[n], prob[n]]) # sample from non-uniform binomial distribution
        #
        labels = []
        data[0, :] = 1
        imgs = []
        #print(fudged_image.shape, np.min(image), np.max(image))
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            color = np.reshape(128.0*np.random.normal(0, 0.1, fudged_image.size), fudged_image.shape)# sampled from Gaussian
            #print('Color:', color.shape, np.min(color), np.max(color))
            temp[:, mask] = fudged_image[:, mask] + color[:, mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds[0])
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds[0])
        return data, np.array(labels)
