import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class CarsClassifier(object):
    def __init__(self, vehicle_data, car_features):
        self.vehicle_data = vehicle_data
        self.car_features = car_features
        self.model = None
        self.scaler = None
        self.test_score = None
        self.save_path = 'classifier_data.pickle'

    def train(self):
        # Load data
        cars = self.vehicle_data.get_cars_images()
        non_cars = self.vehicle_data.get_non_cars_images()

        # Load the images and extract features
        cars_features = self.car_features.extract_features(cars)
        non_cars_features = self.car_features.extract_features(non_cars)

        # Cast things and also create the target variables
        X = np.vstack((cars_features, non_cars_features)).astype(np.float64)
        self.scaler = StandardScaler().fit(X)
        scaled_X = self.scaler.transform(X)

        y = np.hstack((np.ones(len(cars_features)), np.zeros(len(non_cars_features))))

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=0)

        # Initialize and train the model
        self.model = LinearSVC()
        self.model.fit(X_train, y_train)

        # Print the test score
        self.test_score = self.model.score(X_test, y_test)
        print(round(self.test_score, 4))

        # Save everything
        data_to_save = {
            'model': self.model,
            'scaler': self.scaler,
            'test_score': self.test_score
        }

        joblib.dump(data_to_save, self.save_path)

    def load_model(self):
        loaded_data = joblib.load(self.save_path)
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.test_score = loaded_data['test_score']