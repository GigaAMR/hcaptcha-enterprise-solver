from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image, ImageFilter

# Load the trained model
model = load_model('C:/Users/admin/Desktop/opencvtest/pose_detection_model.h5')

# Define the path to the validation directory
validation_dir = 'C:/Users/admin/Desktop/opencvtest/validation'  # Adjust as needed
categories = ['sitting', 'jumping']  # Folder names should match pose labels

# Custom preprocessing function to apply Gaussian blur
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")  # Convert to RGB
    img = img.filter(ImageFilter.GaussianBlur(radius=1))  # Apply Gaussian blur
    img = img.resize((128, 128))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Loop through each category folder
for category in categories:
    category_path = os.path.join(validation_dir, category)
    
    if not os.path.exists(category_path):
        print(f"Path not found: {category_path}")
        continue
    
    print(f"\nPredictions for {category} images:")

    # Loop through each image in the category folder
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # Preprocess the image
        img_array = preprocess_image(img_path)
        
        # Predict the pose
        prediction = model.predict(img_array)

        # Get confidence score for both classes
        confidence_sitting = prediction[0][0]  # Confidence for sitting
        confidence_jumping = 1 - confidence_sitting  # Confidence for jumping

        # Correctly determine the predicted pose based on confidence
        predicted_pose = 'Sitting' if confidence_sitting >= 0.5 else 'Jumping'
        
        # Print the prediction result with confidence
        print(f"{img_name}: Predicted Pose - {predicted_pose}, "
              f"Confidence Sitting: {confidence_sitting:.2f}, "
              f"Confidence Jumping: {confidence_jumping:.2f}")

# Final note: Additional summary could be printed if needed.
