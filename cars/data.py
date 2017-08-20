import os


class VehicleData(object):
    def __init__(self):
        self.cars_loc = './data/vehicles'
        self.non_cars_loc = './data/non-vehicles'

    def get_image_paths(self, images_dir):
        all_images = []
        dirs = os.listdir(images_dir)
        for inner_dir in dirs:
            full_inner_dir = images_dir + '/' + inner_dir
            if not os.path.isdir(full_inner_dir):
                continue

            images = os.listdir(full_inner_dir)
            images = filter(lambda x: x.endswith('.png'), images)
            all_images += list(map(lambda x: full_inner_dir + '/' + x, images))

        return all_images

    def get_cars_images(self):
        return self.get_image_paths(self.cars_loc)

    def get_non_cars_images(self):
        return self.get_image_paths(self.non_cars_loc)
