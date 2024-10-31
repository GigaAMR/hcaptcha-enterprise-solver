# hCaptcha Solver

This project is an image-based hCaptcha solver using a convolutional neural network (CNN) in TensorFlow. It classifies images to identify specific poses, potentially helping solve CAPTCHA challenges by distinguishing between target and non-target poses.

## Project Structure

- **`convert.py`**: Prepares images for training by applying grayscale conversion and Gaussian blur. It processes images in both training and validation folders to enhance model accuracy.

- **`train.py`**: Sets up and trains the CNN model with data augmentation (rotation, shifts, shear, zoom, and flipping) to improve robustness. The model includes convolutional, max pooling, and dense layers, with a binary classification output layer.
  
- **`validate.py`**: Validates the model on a separate dataset. Loads images, applies Gaussian blurring, resizing, and normalization, and uses the model to predict each image's class.

## Features

- **Data Augmentation**: Enhances model generalization by randomly altering images during training.
- **Custom Preprocessing**: Applies Gaussian blur to reduce noise in images before validation.
- **Binary Classification**: Distinguishes between two classes (e.g., "sitting" and "jumping").

TODO:
1, TEST GAN + VAE SYNTHETIC IMAGE GENERATOR

150 epochs 100% accuracy
![image](https://github.com/user-attachments/assets/e4c838d4-6887-40a5-b832-406611d87c00)
