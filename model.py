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

def construct_model_architecture():
    model = Sequential()

    # with lambda normalization
    model.add(Lambda(lambda x: x/255.0, input_shape=SHAPE))
    model.add(Conv2D(filters=8,
                    kernel_size=(2, 2),
                    padding='same',
                    activation='relu'))
    # model.add(Conv2D(filters=8,
    #                 kernel_size=(2, 2),
    #                 padding='same',
    #                 activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=16,
                    kernel_size=(2, 2),
                    padding='same',
                    activation='relu'))
    # model.add(Conv2D(filters=16,
    #                 kernel_size=(2, 2),
    #                 padding='same',
    #                 activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(10, activation='tanh'))
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

    train_datagen = ImageDataGenerator()

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=train,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='other'
    )

    validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=test,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='other'
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
    model.save("model.h5")

def preprocess_data():
    data = pd.read_csv(('train_responses.csv'), names=['id', 'corr'], header=0)
    data["id"] = data["id"].apply(lambda path: path + ".png")
    data['corr'] = data['corr'].astype('float')
    return data

if __name__ == "__main__":
    data = preprocess_data()
    model = construct_model_architecture()
    train(data, model)
