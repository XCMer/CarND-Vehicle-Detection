import matplotlib.image as mpimg
from cars import draw


image_draw = draw.ImageDraw()
img = mpimg.imread('test_images/test1.jpg')
bboxes = [((810, 400), (950, 500))]
img = image_draw.draw_boxes(img, bboxes)

mpimg.imsave('output_images/bounding_box.png', img)
