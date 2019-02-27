import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

lines =[]
correction = 0.2

## Loading data section

print('loading')
with open('data/driving_log.csv') as csvfile: #orginal data
    reader = csv.reader(csvfile)
    for line in reader:
        templine = line
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            templine[i] = 'D:\\Marcin\\Udacity\\SelfDriving cars\\CarND-Behavioral-Cloning-P3-master\\data\\IMG\\' + filename
        lines.append(templine)
    print (len(lines))

print('loading') # 3 laps counterclockwise
with open('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print (len(lines))
    
print('loading') #3 laps clockwise
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print (len(lines))
    
print('loading')  #bridge recovery
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print (len(lines))
        
print('loading')
with open('data6/driving_log.csv') as csvfile: #bridge
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print (len(lines))
       
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.1)

##generator implementation
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                measurements.append(steering_center)
                measurements.append(-1.0*steering_center)
                measurements.append(steering_left)
                measurements.append(-1.0*steering_left)
                measurements.append(steering_right)
                measurements.append(-1.0*steering_right)
                for i in range(3):
                    source_path = batch_sample[i].replace('\\', '/')
                    image= cv2.imread(source_path)
                    images.append(image)
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

print('training')

##Convolution nVidia NN implementation

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger


input_shape = (160, 320, 3)
model = Sequential()
model.add(Cropping2D(cropping=((65,30), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0-0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = "relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
logger = CSVLogger(filename='logs/history.csv')

##run the model    
model.compile(loss='mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples), validation_data=validation_generator, nb_val_samples=6*len(validation_samples), nb_epoch=4, verbose=1, callbacks=[checkpointer, logger])
model.save("model.h5")
print ('saved')
