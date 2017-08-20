import numpy as np
import cv2


class CarsDetector(object):
    def __init__(self, cars_features, classifier, yarea):
        self.cars_features = cars_features
        self.classifier = classifier
        self.ystart = yarea[0]
        self.yend = yarea[1]

        self.window = 64
        self.cells_per_step = 2

    def detect_cars(self, img, scale=1, rescale=True):
        # We train on PNGs (scaled 0 to 1), but we get JPEGs for finding
        # cars. We scale the JPEG's 255 down to 1 again.
        img = np.copy(img)
        if rescale:
            img = img.astype(np.float32) / 255.0

        # STEP 1: Truncate the image to the area we want to search
        img_to_search = img[self.ystart:self.yend, :, :]

        # STEP 2: Convert to the color-space that we want
        img_to_search = self.cars_features.rgb_to_cspace(img_to_search)

        # STEP 3: Rescale
        if scale != 1:
            img_shape = img_to_search.shape
            img_to_search = cv2.resize(img_to_search, (np.int(img_shape[1] / scale), np.int(img_shape[0] / scale)))

        # STEP 4: Calculate block sizes
        img_shape = img_to_search.shape
        nx_blocks = (img_shape[1] // self.cars_features.pixel_per_cell) - self.cars_features.cell_per_block + 1
        ny_blocks = (img_shape[0] // self.cars_features.pixel_per_cell) - self.cars_features.cell_per_block + 1
        n_features_per_block = self.cars_features.orient * self.cars_features.cell_per_block ** 2

        # STEP 5: Calculate steps
        n_blocks_per_window = (self.window // self.cars_features.pixel_per_cell) - self.cars_features.cell_per_block + 1
        nx_steps = (nx_blocks - n_blocks_per_window) // self.cells_per_step
        ny_steps = (ny_blocks - n_blocks_per_window) // self.cells_per_step

        # STEP 6: Calculate HOG features
        hog_features = []
        for i in range(img_to_search.shape[2]):
            hog = self.cars_features.get_hog_features(img_to_search[:,:,i], vis=False, feature_vector=False)
            hog_features.append(hog)

        # STEP 7: Sliding window
        bboxes = []
        for xb in range(nx_steps):
            for yb in range(ny_steps):
                ypos = yb * self.cells_per_step
                xpos = xb * self.cells_per_step

                # Extract HOG for this patch
                patch_hog_features = []
                for hog in hog_features:
                    patch_hog = hog[ypos:ypos + n_blocks_per_window, xpos:xpos + n_blocks_per_window].ravel()
                    patch_hog_features.append(patch_hog)

                patch_hog_features = np.hstack(patch_hog_features)

                xleft = xpos * self.cars_features.pixel_per_cell
                ytop = ypos * self.cars_features.pixel_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_to_search[ytop:ytop + self.window, xleft:xleft + self.window], (64, 64))

                # Get color features
                # spatial_features = bin_spatial(subimg, size=spatial_size)
                # hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = self.classifier.scaler.transform(
                    np.hstack((patch_hog_features,)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.classifier.model.predict(test_features)

                # Add boxes
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(self.window * scale)
                    bboxes.append(((xbox_left, ytop_draw + self.ystart), (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)))

        return bboxes
