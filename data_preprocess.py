import numpy as np
import cv2
import os
import keras

data_path = "training_data/aaacpaqd.png"

def process_image(img):
    image = cv2.imread(str(img), 1)
    image = image[2:-21, 21:-2, :]  # crop image from 150x150 to 127x127
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114]) # convert rgb to grayscale
    return image

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def convert_images():
    path1 = "training_data/"
    path2 = "processed_data/"
    data_file_paths = os.listdir(path1)
    for file in data_file_paths:
        processed_image = process_image(path1 + file)
        save_image(path2 + file, processed_image)

if __name__ == "__main__":
    print("training_data/aaacpaqd.png")
    processed_image = process_image("training_data/aaacpaqd.png")
    save_image('processed_data/aaacpaqd.png', processed_image)
    # convert_images()

"""
    create
"""
