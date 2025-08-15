import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#####################################################
#####################################################
#####################################################

def balance_steering_angles(images, steering_angles):
    straight_threshold = 0.05 
    straight_keep_ratio = 0.2  # only 20% straights
    
    images = np.array(images)
    steering_angles = np.array(steering_angles)

    total_size = len(images)
    straight_indices = []
    turning_indices = []

    for i, angle in enumerate(steering_angles):
        if abs(angle) <= straight_threshold:
            straight_indices.append(i)
        else:
            turning_indices.append(i)

    keep_straight = np.random.choice(
        straight_indices,
        size=int(len(straight_indices) * straight_keep_ratio),
        replace=False
    )

    reduced_set = np.concatenate([keep_straight, turning_indices])

    if len(reduced_set) < total_size:
        extra_needed = total_size - len(reduced_set)
        extra_indices = np.random.choice(reduced_set, size=extra_needed, replace=True)
        final_indices = np.concatenate([reduced_set, extra_indices])
    else:
        final_indices = reduced_set

    np.random.shuffle(final_indices)
    return images[final_indices], steering_angles[final_indices]


#####################################################
#####################################################
#####################################################

def augment_image(image, steering_angle):

    augmented_image = image.copy()
    augmented_angle = steering_angle

    if random.random() > 0.5:
        augmented_image = cv2.flip(augmented_image, 1)
        augmented_angle = -steering_angle

    if random.random() > 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        augmented_image = np.clip(augmented_image * brightness_factor, 0, 1)


    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = augmented_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))
    
    if random.random() > 0.5:
        tx = random.uniform(-50, 50)  
        h, w = augmented_image.shape[:2]
        augmented_angle += tx * 0.002  #adjust steering slightly
        M = np.float32([[1, 0, tx], [0, 1, 0]])
        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))

    return augmented_image, augmented_angle

#####################################################
#####################################################
#####################################################

# https://medium.com/analytics-vidhya/train-keras-model-with-large-dataset-batch-training-6b3099fdf366
def batch_generator(X, y, batch_size=32):
    num_samples = len(X)
    while True: 
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset+batch_size]
            batch_images = []
            batch_angles = []
            for idx in batch_indices:
                img = X[idx]
                angle = y[idx]
                if np.random.rand() < 0.5:
                    img, angle = augment_image(img, angle)
                batch_images.append(img)
                batch_angles.append(angle)
            yield np.array(batch_images), np.array(batch_angles)



#####################################################
#####################################################
#####################################################

def preprocess_data():

    data_path = './data/driving_log.csv'

    data = pd.read_csv(data_path, header=None)
    images = data[0].values
    steering_angles = data[3].values
    
    plt.figure(figsize=(10, 6))
    plt.hist(steering_angles, bins=30)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.title('Steering Angle Distribution')
    plt.show()
    
    images, steering_angles = balance_steering_angles(images, steering_angles)

    plt.figure(figsize=(10, 6))
    plt.hist(steering_angles, bins=30)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.title('Steering Angle Distribution')
    plt.show()

    processed_images = []
    valid_steering_angles = []
    
    for i, img_path in enumerate(images):
        if i % 1000 == 0:
            print(f"[INFO] Processed {i}/{len(images)} images")

        filename = os.path.basename(img_path)
        full_path = os.path.join("data", "IMG", filename)
        
        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.resize(img, (200, 66))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = img / 255
                # processed_img = img
                if img is not None:
                    processed_images.append(img)
                    valid_steering_angles.append(steering_angles[i])
        else:
            print(f"processed image is None for {img_path}")

    X = np.array(processed_images)
    y = np.array(valid_steering_angles)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training set shape: {X_train.shape}")
    print(f"[INFO] Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test



#####################################################
#####################################################
#####################################################



def model_training(X_train, X_test, y_train, y_test, batch_size=32):

    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(1164, activation='relu'),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    train_gen = batch_generator(X_train, y_train, batch_size)

    steps_per_epoch = len(X_train) // batch_size

    # https://medium.com/analytics-vidhya/train-keras-model-with-large-dataset-batch-training-6b3099fdf366
    # fit_generator is deprecated so using fit
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        epochs=10
    )
    model.save('modelv16.h5')


    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return model


#####################################################
#####################################################
#####################################################



def main():

    X_train, X_test, y_train, y_test = preprocess_data()
    model = model_training(X_train, X_test, y_train, y_test)

    return model


if __name__ == "__main__":
    main()
