""" import libraries """
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#import tensorflow as tf
#from tensorflow.keras import models, datasets, layers

# shape of input data
SHAPE = (127, 127, 1)
BATCH_SIZE = 200
INPUT_SHAPE = (BATCH_SIZE, SHAPE)

def model_architecture():
    model = Sequential()

    # with lambda normalization
    model.add(Lambda(lambda x: x/255.0, input_shape=SHAPE))
    model.add(Conv2D(24, (3, 3), activation='relu'))

    #without lambda normalization
    #model.add(Conv2D(24, (3, 3), activation='relu', input_shape=SHAPE))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(36, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1))
    model.summary()
    return model

def train(data_df, model, validation_percentage=0.2):

    # create checkpoint to save model after ever epoch
    checkpoint = ModelCheckpoint(
            'model-{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='auto')
    model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy'])

    train, test = train_test_split(data_df, test_size=0.2)

    train_datagen = ImageDataGenerator(
        rescale=1./255)

    validation_datagen = ImageDataGenerator(
        rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=train,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='sparse'
    )

    validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=test,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='sparse'
    )

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=10,
            verbose=1,
            callbacks=[checkpoint],
            validation_data=validation_generator,
            validation_steps=200,
            use_multiprocessing=False
            )

if __name__ == "__main__":
    model = model_architecture()

    data = pd.read_csv(('train_responses.csv'), names=['id', 'corr'])
    data["id"] = data["id"].apply(lambda path: path + ".png")
    train(data, model)
    model.save("model.h5")
