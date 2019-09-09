import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# shape of input data
SHAPE = (127, 127, 1)
BATCH_SIZE = 200
INPUT_SHAPE = (BATCH_SIZE, SHAPE)

def construct_model_architecture():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0, input_shape=SHAPE))
    model.add(Conv2D(filters=8,
                    kernel_size=(2, 2),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=16,
                    kernel_size=(2, 2),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))
    model.summary()
    return model

def get_generator(data):
    data = data.sample(len(data), random_state=0)
    train, test = train_test_split(data, test_size=0.2)      # randomly select test dataset

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_dataframe(
            dataframe=train,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='other'
    )

    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=test,
            directory='processed_data',
            x_col='id',
            y_col='corr',
            target_size=(127, 127),
            color_mode='grayscale',
            class_mode='other'
    )

    return train_generator, validation_generator

def train(data, model):
    train_generator, validation_generator = get_generator(data)

    save_checkpoint = ModelCheckpoint(
            'model-{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min')

    early_stoppage = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            mode='min',
            patience=2)

    adam = optimizers.Adam(lr=0.0001)
    model.compile(
            optimizer=adam,
            loss='mean_absolute_error')

    model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=10,
            verbose=1,
            callbacks=[save_checkpoint, early_stoppage],
            validation_data=validation_generator,
            validation_steps=200
            )

def preprocess_data():
    data = pd.read_csv(('train_responses.csv'), names=['id', 'corr'], header=0)
    data["id"] = data["id"].apply(lambda path: path + ".png")
    data['corr'] = data['corr'].astype('float')
    return data

if __name__ == "__main__":
    data = preprocess_data()
    model = construct_model_architecture()
    train(data, model)
