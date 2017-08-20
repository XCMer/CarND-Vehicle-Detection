import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog
from tqdm import tqdm


class CarFeatures(object):
    def __init__(self, cspace='RGB',
                 hist_nbins=32, hist_bins_range=(0,256),
                 spatial_bin_size=(32,32),
                 orient=9, pixel_per_cell=8, cell_per_block=2):

        # Color space
        self.cspace = cspace

        # Color histogram
        self.hist_bins_range = hist_bins_range
        self.hist_nbins = hist_nbins

        # Spatial binning
        self.spatial_bin_size = spatial_bin_size

        # HOG
        self.cell_per_block = cell_per_block
        self.pixel_per_cell = pixel_per_cell
        self.orient = orient

        self.cspaces = {
            'HSV': cv2.COLOR_RGB2HSV,
            'LUV': cv2.COLOR_RGB2LUV,
            'HLS': cv2.COLOR_RGB2HLS,
            'YUV': cv2.COLOR_RGB2YUV,
            'GRAY': cv2.COLOR_RGB2GRAY,
            'YCrCb': cv2.COLOR_RGB2YCrCb
        }

    def rgb_to_cspace(self, img):
        if self.cspace != 'RGB':
            return cv2.cvtColor(img, self.cspaces[self.cspace])
        else:
            return np.copy(img)

    def get_color_histogram(self, img):
        # Compute the histogram of each channel separately
        hists = []
        for i in range(img.shape[2]):
            hist = np.histogram(img[:, :, i], bins=self.hist_nbins, range=self.hist_bins_range)
            hists.append(hist)

        # Find bin centers
        bin_edges = hists[0][1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate(list(map(lambda x: x[0], hists)))

        # Return the individual histograms, bin_centers and feature vector
        return tuple(hists) + (bin_centers, hist_features)

    def get_bin_spatial(self, img):
        feature_img = np.copy(img)
        features = cv2.resize(feature_img, self.spatial_bin_size).ravel()

        return features

    def get_hog_features(self, img, vis=False, feature_vector=True):
        return hog(img, orientations=self.orient, pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
                   cells_per_block=(self.cell_per_block, self.cell_per_block), visualise=vis,
                   feature_vector=feature_vector)

    def extract_features(self, imgs):
        features = []
        for img in tqdm(imgs):
            img = mpimg.imread(img)
            img = self.rgb_to_cspace(img)
            channel_features = []
            for i in range(img.shape[2]):
                channel_features.append(self.get_hog_features(img[:, :, i], vis=False, feature_vector=True))

            features.append(np.ravel(channel_features))

        return features
