from photoelasticity.image_detection import extract_circle_and_count_stripes

if __name__=='__main__':
    # Usage
    image_path = 'path/to/your/photoelasticity_image.jpg'
    stripe_count, circular_image = extract_circle_and_count_stripes(image_path)
    print(f"Number of photoelasticity stripes: {stripe_count}")
