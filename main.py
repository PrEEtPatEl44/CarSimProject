import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


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
                img = img / 255.0
                processed_image = img

        if processed_image is not None:
            processed_images.append(processed_image)
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



def model_training(X_train, X_test, y_train, y_test):

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
    history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    model.save('model.h5')


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
