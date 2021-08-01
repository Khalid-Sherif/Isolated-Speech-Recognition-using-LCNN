'''
Isolated Speech Recognition Using LCNN
Author: Khaled Mohamed
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import os
#from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.metrics import classification_report


def wav2mfcc(file_path, augment = False, max_pad_len=13):
    wave, sr = librosa.load(file_path, mono=True, sr=8000, duration = 1.024)
    
    if augment == True:
        bins_per_octave = 12
        pitch_pm = 4
        pitch_change =  pitch_pm * 2*(np.random.uniform())   
        wave = librosa.effects.pitch_shift(wave, 
                                          8000, n_steps=pitch_change, 
                                          bins_per_octave=bins_per_octave)
        
        speed_change = np.random.uniform(low=0.9,high=1.1)
        wave = librosa.effects.time_stretch(wave, speed_change)
        wave = wave[:8192]

    duration = wave.shape[0]/sr
    speed_change = 2.0* duration/1.024
    wave = librosa.effects.time_stretch(wave, speed_change)
    wave = wave[:4096]
    
    wave = librosa.util.normalize(wave)
    mfcc = librosa.stft(wave, hop_length=int(0.048*sr), n_fft=int(0.096*sr))
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    #print("shape=",mfcc.shape[1], wave.shape[0])
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #mfcc = mfcc[2:24,:]
    return mfcc, duration, sr

def get_data(dir = '', augment= False):
    labels = []
    mfccs = []
    durations = []
    sampling_rates = []
    filenames = []

    for f in tqdm(os.listdir(dir)):
        if f.endswith('.wav'):
            mfcc, duration, sr = wav2mfcc(dir + "/" + f, augment)
            mfccs.append(mfcc)
            durations.append(duration)
            sampling_rates.append(sr)
            # List of labels
            label = f.split('_')[0]
            labels.append(label)
            filenames.append(dir + "/" + f)
    return filenames, np.asarray(mfccs), np.asarray(durations), np.asarray(sampling_rates), tf.keras.utils.to_categorical(labels), labels

def mfm(x):
    shape = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
    x1 = tf.keras.layers.Cropping2D(cropping=((0, shape[3] // 2), 0))(x)
    x2 = tf.keras.layers.Cropping2D(cropping=((shape[3] // 2, 0), 0))(x)
    x = tf.keras.layers.Maximum()([x1, x2])
    x = tf.keras.layers.Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
    #x = tf.keras.layers.Reshape([shape[1], shape[2], shape[3] // 2])(x)
    return x


'''
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3, figsize=(15,15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    max = np.max(images)
    min = np.min(images)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        #ax.imshow(images[i].reshape(img_shape), cmap='binary')
        im = librosa.display.specshow(images[i], ax=ax, vmin=min, vmax=max)

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_images(mfccs[100:109], cls_true[100:109])

n, bins, patches = plt.hist(durations, bins=100)
'''


filenames, mfccs, durations, sampling_rates, labels, cls_true = get_data('recordings2', augment = False)
    
filenames_a, mfccs_a, durations_a, sampling_rates_a, labels_a, cls_true_a = get_data('recordings2', augment = True)

mfccs = np.append(mfccs, mfccs_a, axis=0)
labels = np.append(labels, labels_a, axis =0)
    
dim_1 = mfccs.shape[1]
dim_2 = mfccs.shape[2]
channels = 1
classes = 10
    
print("sampling rate (max) = ", np.max(sampling_rates))
print("sampling rate (min) = ", np.min(sampling_rates))
print("duration (max) = ", np.max(durations))
print("duration (avg) = ", np.average(durations))
print("duration (min) = ", np.min(durations))
print("mffc matrix = ", mfccs.shape)

X = mfccs
X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
y = labels

input_shape = (dim_1, dim_2, channels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#model = get_cnn_model(input_shape, classes)
model = tf.keras.models.Sequential()

# 1st layer
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=input_shape, activation= lambda mfm:mfm))
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Lambda(mfm))
#model.add(tf.keras.layers.MaxPooling2D(pool_size= 3, strides= 2, padding= 'same'))

# 2nd layer
#model.add(tf.keras.layers.Conv2D(48, kernel_size=(2,2),activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Lambda(mfm))
#model.add(tf.keras.layers.MaxPooling2D(pool_size= 3, strides= 2, padding= 'same'))

#model.add(tf.keras.layers.MaxPooling2D(pool_size= (3,3)))
#model.add(tf.keras.layers.Dropout(0.5))

# 3rd layer
#model.add(tf.keras.layers.Conv2D(120, kernel_size=(2,2), activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Lambda(mfm))

# 4th layer
#model.add(tf.keras.layers.Conv2D(208, kernel_size=(3,3), activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())

# 5th layer
#model.add(tf.keras.layers.Conv2D(400, kernel_size=(2,2), activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())


model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())

#model.add(tf.keras.layers.Dense(256, activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.4))
#model.add(tf.keras.layers.Dense(128, activation= lambda mfm:mfm))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(64, activation= lambda mfm:mfm))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(classes, activation= 'softmax'))
#model.add(tf.keras.layers.Activation('softmax'))


model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", factor= 0.2, mode= "auto")
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= tf.keras.optimizers.Adam(), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(X_train, y_train, batch_size= 32, epochs= 30, verbose= 1, validation_split= 0.1, callbacks= [reduce_lr])
# X_train, X_test, y_train, y_test, cnn_model = get_all()

#print(cnn_model.summary())

#cnn_model.fit(X_train, y_train, batch_size=64, epochs=15, verbose=1, validation_split=0.1)

predictions = model.predict_classes(X_test)
print(classification_report(y_test, tf.keras.utils.to_categorical(predictions)))


'''
# Used to check if the framework is working properly or not

def get_cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model
'''
