from photoelasticity.data import get_day_data
from photoelasticity.data_processing import process_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes

if __name__ == '__main__':
    # Usage
    related_data = [
        [247, 259, 263, 275, 276],
        [248, 258, 264, 274, 277],
        [249, 257, 265, 273, 278],
        [250, 256, 266, 272, 279],
        [251, 255, 267, 271, 280],
        [252, 254, 268, 270, 281]
    ]
    interesting_images = [
        [f"V_0{image_num}.jpg"
         for image_num in bundle]
        for bundle in related_data
    ]
    related_data = [
        [extract_circle_and_count_stripes(image_path)
         for image_path in get_day_data(1, interesting_images_bundle)]
        for interesting_images_bundle in interesting_images
    ]
    processed_data = [
        [process_data(data) for data in bundle if data is not None]
        for bundle in related_data
    ]
    # flat_data = [x for y in processed_data for x in y[2:]]
    # for i, data in enumerate(flat_data):
    #     find_fit_params(data, f"data of {i}")
# for i, data in enumerate(processed_data):
#     data = [subdata for subdata in data if len(subdata) > 2000]
#     min_len = min(len(subdata) for subdata in data)
#     min_rad = min_len // 2
#     truncated_data = [subdata[len(subdata) - min_rad:len(subdata) + min_rad] for subdata in data]
#     average_data = np.mean(truncated_data, axis=0)
#     find_fit_params(average_data, f"data of {i}")
