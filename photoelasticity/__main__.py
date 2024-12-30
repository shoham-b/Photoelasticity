from photoelasticity.data import get_day_data
from photoelasticity.image_detection import extract_circle_and_count_stripes

if __name__=='__main__':
    # Usage
    extract_circle_and_count_stripes("C:\\Users\\shoha\\PycharmProjects\\Photoelasticity\\data\\day1\\V_0275.jpg")
    # for image_path in list(get_day_data(1))[3:]:
    #     extract_circle_and_count_stripes(image_path)
    #     break