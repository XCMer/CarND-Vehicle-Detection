from cars import data


vehicle_data = data.VehicleData()
total_cars = len(vehicle_data.get_cars_images())
total_non_cars = len(vehicle_data.get_non_cars_images())
total = total_cars + total_non_cars

print("Vehicle images: ", total_cars, str(round(total_cars / total * 100.0, 2)) + '%')
print("Non-vehicle images: ", total_non_cars, str(round(total_non_cars / total * 100.0, 2)) + '%')
