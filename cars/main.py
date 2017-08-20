import sys
import matplotlib.image as mpimg
from cars import classifier, data, features, detector, draw


cars_data = data.VehicleData()
cars_features = features.CarFeatures('YCrCb')
cars_classifier = classifier.CarsClassifier(cars_data, cars_features)
cars_detector = detector.CarsDetector(cars_features, cars_classifier, (400, 656))
image_draw = draw.ImageDraw()

cars_classifier.load_model()

cmd = sys.argv[1]
if cmd == 'train':
    cars_classifier.train()
elif cmd == 'detect_image':
    img = mpimg.imread(sys.argv[2])
    bboxes = cars_detector.detect_cars(img, 1.5, rescale=True)
    draw_img = image_draw.draw_boxes(img, bboxes)

    mpimg.imsave('output_images/' + sys.argv[3], draw_img)
