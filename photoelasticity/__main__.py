from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes

if __name__ == '__main__':
    # Usage
    interesting_images = [f"V_0{image_num}.jpg" for image_num in range(254, 259)]
    # extract_circle_and_count_stripes("C:\\Users\\shoha\\PycharmProjects\\Photoelasticity\\data\\day1\\V_0275.jpg")
    all_interesting_files = list(get_day_data(1, interesting_images))
    for image_path in all_interesting_files:
        data, radius = extract_circle_and_count_stripes(image_path)
        find_fit_params(data, radius, image_path.name)
