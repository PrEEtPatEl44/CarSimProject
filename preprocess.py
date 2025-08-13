import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

data = pd.read_csv('./data/driving_log.csv', header=None)
images = data[0].values
steering_angles = data[3].values

print("Showing histogram")
plt.hist(steering_angles, bins=30)
plt.xlabel('Steering Angle')
plt.ylabel('frequency ')
plt.show()


def preprocess_image(image_path):
    filename = os.path.basename(image_path)
    full_path = os.path.join("data", "IMG", filename)
    
    if os.path.exists(full_path):
        try:

            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.resize(img, (200, 66))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = img / 255.0
                return img
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
    return None


processed_images = []
valid_steering_angles = []

for i, img_path in enumerate(images):
    # print(i)
    if i % 1000 == 0:
        print(f"[INFO]: Processed {i}/{len(images)} ")
        
    
    processed_img = preprocess_image(img_path)
    if processed_img is not None:
        processed_images.append(processed_img)
        valid_steering_angles.append(steering_angles[i])

X = np.array(processed_images)
y = np.array(valid_steering_angles)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")