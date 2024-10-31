from PIL import Image, ImageFilter
import os

def apply_gaussian_blur(input_folder):
    # Iterate over all subdirectories in the input folder
    for subdir, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more formats if needed
                img_path = os.path.join(subdir, filename)
                # Open the image
                with Image.open(img_path) as img:
                    # Convert to grayscale
                    gray_image = img.convert("L")
                    # Apply Gaussian blur with radius 2
                    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))
                    # Save the new blurred image, overwriting the old one
                    blurred_image.save(img_path)

def main():
    # Define the base directory for training and validation
    base_dir = r'C:\Users\admin\Desktop\opencvtest\src'
    training_dir = os.path.join(base_dir, 'training')
    validation_dir = os.path.join(base_dir, 'validation')

    # Apply Gaussian blur to images in training and validation folders
    apply_gaussian_blur(training_dir)
    apply_gaussian_blur(validation_dir)

if __name__ == "__main__":
    main()
