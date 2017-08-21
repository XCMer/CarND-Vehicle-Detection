import sys
import matplotlib.image as mpimg
from tqdm import tqdm
from cars import classifier, data, features, detector, draw
from moviepy.editor import VideoFileClip


cars_data = data.VehicleData()
cars_features = features.CarFeatures('YCrCb', hist_nbins=64)
cars_classifier = classifier.CarsClassifier(cars_data, cars_features)
cars_detector = detector.CarsDetector(cars_features, cars_classifier, (400, 656))
image_draw = draw.ImageDraw()

video_cars_detector = detector.VideoCarsDetector(cars_detector, image_draw, (1280, 728),
                                                 scales=(0.75, 1.0, 1.25, 1.5),
                                                 save_heatmaps_to_file=False,
                                                 load_heatmaps_from_file=False,
                                                 save_heatmap_video=False)

cars_classifier.load_model()

cmd = sys.argv[1]
if cmd == 'train':
    cars_classifier.train()
elif cmd == 'detect_image':
    for i in tqdm(range(1, 7)):
        img = mpimg.imread('test_images/test' + str(i) + '.jpg')
        bboxes = cars_detector.detect_cars(img, 1.5, rescale=True)
        draw_img = image_draw.draw_boxes(img, bboxes, color=(255, 0, 0))

        bboxes = cars_detector.detect_cars(img, 1.25, rescale=True)
        draw_img = image_draw.draw_boxes(draw_img, bboxes, color=(0, 255, 0))

        bboxes = cars_detector.detect_cars(img, 1.0, rescale=True)
        draw_img = image_draw.draw_boxes(draw_img, bboxes, color=(0, 0, 255))

        bboxes = cars_detector.detect_cars(img, 0.75, rescale=True)
        draw_img = image_draw.draw_boxes(draw_img, bboxes, color=(255, 255, 0))

        mpimg.imsave('output_images/test' + str(i) + '_detect.png', draw_img)
elif cmd == 'detect_video':
    input_filename = sys.argv[2]
    output_filename = input_filename[:-4] + '_output.mp4'

    video_clip = VideoFileClip(input_filename)#.subclip(30, 35)
    output_clip = video_clip.fl_image(video_cars_detector.detect_cars_in_frame)
    output_clip.write_videofile(output_filename, audio=False)

    video_cars_detector.save_heatmaps()
