import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, BatchNormalization,Dense,Dropout,Activation,Flatten,SimpleRNN, Conv1D, MaxPooling1D, Add
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical, Sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt



#Reading the labels in the folder one by one
def labeling(path):
    # saving the label of each sample
    x = [path+item  for item in os.listdir(path)] 
    y = [item.split('_')[1] for item in os.listdir(path)]
    #splitting our database randomly
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=8)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.50, random_state=8)
    return x_train, x_test, x_val, y_train, y_test, y_val
   
path = './babbleoriginal/splitted/'
x_train, x_test, x_val, y_train, y_test, y_val = labeling(path)
np.savetxt('x_train.out', x_train, fmt='%s') 
np.savetxt('x_test.out', x_test, fmt='%s') 
np.savetxt('x_val.out', x_val, fmt='%s') 
np.savetxt('y_train.out', y_train, fmt='%s')
np.savetxt('y_test.out', y_test, fmt='%s')
np.savetxt('y_val.out', y_val, fmt='%s')

print('loading...')

X_train = np.loadtxt('x_train.out', dtype='str') 
X_test = np.loadtxt('x_test.out', dtype='str') 
X_val = np.loadtxt('x_val.out', dtype='str') 
Y_train = np.loadtxt('y_train.out', dtype='str')
Y_train = pd.DataFrame(Y_train,columns=['class'])
Y_train = np.array(Y_train['class'])
#categorial
Y_train = np.array(pd.get_dummies(Y_train))
Y_test = np.loadtxt('y_test.out', dtype='str')
Y_test = pd.DataFrame(Y_test,columns=['class'])
Y_test = np.array(Y_test['class'])
#categorial
Y_test = np.array(pd.get_dummies(Y_test))
Y_val = np.loadtxt('y_val.out', dtype='str')
Y_val = pd.DataFrame(Y_val,columns=['class'])
Y_val = np.array(Y_val['class'])
#categorial
Y_val = np.array(pd.get_dummies(Y_val))

print(len(X_train),'X_train')
print(len(X_test),'X_test')
print(len(X_val),'X_val')



# reading the files one by one
def readfile1(x):
    F = []
    for item in  x:
        # sampling rate: 22500 Hz
        data, sr = librosa.load(item)
        # sound source seperation technique
        y_harm, y_perc = librosa.effects.hpss(data)
        # mfccs 
        lfccs = librosa.feature.mfcc(y=y_harm, sr=sr, n_mfcc=120)
        # mean values
        lfccs = np.mean(lfccs.T,axis=0)
        #mfcc
        lfccs1 = librosa.feature.mfcc(y=y_perc, sr=sr, n_mfcc=120)
        #mean values
        lfccs1 = np.mean(lfccs1.T,axis=0)
        # stacking the extracted feature vectors
        lfccs2 = np.vstack((lfccs, lfccs1)).T
        F.append(lfccs2)
    return np.array(F)
    

# generating data for reading them batch by batch

class My_Custom_Generator(Sequence) :
    def __init__(self, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        return (np.array(readfile1(batch_x)), batch_y)
    
batch_size = 52

my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_test, Y_test, batch_size)



# CNN
model = Sequential()    
model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu", input_shape=(120,2))) 
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu"))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu"))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu"))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(64,activation = "sigmoid"))
model.add(Dense(7))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss= "categorical_crossentropy", 
optimizer = Adam(lr=0.005), metrics = ['accuracy'])


filepath='saved_models/1.hdf5'
# Check whether the specified path exists or not
isExist = os.path.exists(filepath)
if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(filepath)
    
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

history = model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(len(X_train) // batch_size),
                   epochs = 5,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(len(X_val) // batch_size))


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Saving weights
model.save("model.h5")
print("Saved model to disk")

Y_pred = model.predict(readfile1(X_test))

matrix = metrics.confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
f1score = metrics.f1_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1),average = 'macro')

class_names = ["FLRoom", "FMRoom","FOffice","SLRoom","SMRoom","SOffice","Lobby"]
df = pd.DataFrame(matrix, index=class_names, columns=class_names)
figsize = (10,7)
fontsize=14
fig = plt.figure(figsize=figsize)
heatmap = sns.heatmap(df, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix: Multi-Class Classification')
fig.savefig('confusion matrix.png')

