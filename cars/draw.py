import cv2
import numpy as np


class ImageDraw(object):
    def draw_boxes(self, img, bboxes, color=(255, 0, 0), thick=6):
        draw_img = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

        return draw_img
