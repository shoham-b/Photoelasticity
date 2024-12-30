from photoelasticity.data import get_day_data
from photoelasticity.image_detection import extract_circle_and_count_stripes

if __name__=='__main__':
    # Usage
    for image_path in get_day_data(1):
        extract_circle_and_count_stripes(image_path)
