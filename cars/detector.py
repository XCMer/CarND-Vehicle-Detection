import numpy as np
import cv2
from scipy.ndimage.measurements import label
import collections
import pickle


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
                spatial_features = self.cars_features.get_bin_spatial(subimg)
                hist_features = self.cars_features.get_color_histogram(subimg)[-1]

                # Scale features and make a prediction
                test_features = self.classifier.scaler.transform(
                    np.concatenate((patch_hog_features, hist_features, spatial_features)).reshape(1, -1))
                # test_features = self.classifier.scaler.transform(np.hstack((hist_features, spatial_features)).reshape(1, -1))
                test_prediction = self.classifier.model.predict(test_features)

                # Add boxes
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(self.window * scale)
                    bboxes.append(((xbox_left, ytop_draw + self.ystart), (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)))

        return bboxes


class VideoCarsDetector(object):
    def __init__(self, cars_detector, image_draw, shape, scales=(1.5,),
                 save_heatmaps_to_file=False, load_heatmaps_from_file=False, save_heatmap_video=False):
        self.cars_detector = cars_detector
        self.image_draw = image_draw
        self.shape = shape
        self.savable_heatmaps = []
        self.savable_heatmaps_current = 0
        self.heatmaps = collections.deque(maxlen=10)
        self.heatmap = None
        self.save_heatmaps_to_file = save_heatmaps_to_file
        self.load_heatmaps_from_file = load_heatmaps_from_file
        self.save_heatmap_video = save_heatmap_video
        self.scales = scales

        # Load heatmaps from file
        if self.load_heatmaps_from_file:
            self.load_heatmaps()

    def detect_cars_in_frame(self, img):
        if self.load_heatmaps_from_file:
            self.heatmaps.append(self.savable_heatmaps[self.savable_heatmaps_current])
            self.savable_heatmaps_current += 1
        else:
            # STEP 1: Get bounding boxes
            bboxes = []
            for scale in self.scales:
                bboxes += self.cars_detector.detect_cars(img, scale=scale)

            # STEP 2: Add heatmap of the current frame
            self.add_heatmap(bboxes)

        # STEP 3: Update heatmap aggregate with thresholding
        # applied to the last `n` heatmaps
        self.update_heatmap()

        # STEP 4: Put bounding boxes
        output_bboxes = self.get_bounding_boxes()

        # STEP 5: Draw boxes
        if self.save_heatmap_video:
            output_img = np.clip(255.0 * self.heatmap, 0, 255)
            output_img = cv2.merge([output_img, output_img, output_img])
        else:
            output_img = self.image_draw.draw_boxes(img, output_bboxes)

        return output_img

    def add_heatmap(self, bboxes):
        heatmap = np.zeros((self.shape[1], self.shape[0])).astype(np.float)
        for box in bboxes:
            x1 = box[0][1]
            x2 = box[1][1]
            y1 = box[0][0]
            y2 = box[1][0]

            heatmap[x1:x2, y1:y2] += 1

        self.heatmaps.append(heatmap)

        if self.save_heatmaps_to_file:
            self.savable_heatmaps.append(heatmap)

    def update_heatmap(self):
        self.heatmap = np.sum(self.heatmaps, axis=0)

        self.heatmap[self.heatmap <= 6.0 * 2.5] = 0

    def get_bounding_boxes(self):
        labels = label(self.heatmap)
        bboxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            bboxes.append(bbox)

        return bboxes

    def save_heatmaps(self):
        if self.save_heatmaps_to_file:
            with open('heatmaps.pickle', 'wb') as f:
                pickle.dump(self.savable_heatmaps, f)

    def load_heatmaps(self):
        with open('heatmaps.pickle', 'rb') as f:
            self.savable_heatmaps = pickle.load(f)
            self.savable_heatmaps_current = 0
