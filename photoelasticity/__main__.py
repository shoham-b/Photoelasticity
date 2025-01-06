from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes
from photoelasticity.data_processing import strip_dark_boundaries, process_data

if __name__ == '__main__':
    # Usage
    interesting_images = [
        f"V_0{image_num}.jpg"
        for image_num in range(249, 282)
        if image_num != 260
    ]
    all_interesting_files = list(get_day_data(1, interesting_images))
    for image_path in all_interesting_files:
        data = extract_circle_and_count_stripes(image_path)
        if image_path.name == "V_0256.jpg":
            data = process_data(data)
            find_fit_params(data, image_path.name, 9)
        elif image_path.name == "V_0257.jpg":
            data = process_data(data, 0.4, center_max=True)
            find_fit_params(data, image_path.name, 8)
        elif image_path.name == "V_0258.jpg":
            data = process_data(data, 0.4)
            find_fit_params(data, image_path.name, 4)
        else:
            data = (
                process_data(data))

            find_fit_params(data, image_path.name)
