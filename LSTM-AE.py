# %%
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler

import json
from Config.helpers import PyJSON

with open("Config/config.json", "r") as f:
    CONFIG = PyJSON(json.load(f))
f.close()

## Load the data from the CSV file
train_data = pd.read_csv('./Dataset/train7.csv')
test_data = pd.read_csv('./Dataset/DriveCycle10-test-100.csv')
nom_data = pd.read_csv('./Dataset/test.csv')        # To check the performance of ML model


def FeatureSelection(drop_column, data):
    return data.drop(drop_column, axis=1)      # To check the performance of ML model

timesteps = 25   # dataset have 100 ms sample rate for each sec we create 10 dataset a input

def Preprocessing(train_data, test_data, nom_data):
    #normalise test/train
    for i in range(train_data.shape[1]):
        for j in range(test_data.shape[1]):
            if train_data.columns[i] == test_data.columns[j]:
                maxi = train_data.iloc[:,i].max()
                mini = train_data.iloc[:,i].min()   #,test_data.iloc[:,i].max())
                train_data.iloc[:,i] = (train_data.iloc[:,i]-mini)/(maxi-mini)
                test_data.iloc[:,j] = (test_data.iloc[:,j]-mini)/(maxi-mini)
                nom_data.iloc[:,j] = (nom_data.iloc[:,j]-mini)/(maxi-mini)
    
    #Freature Selection
    drop_column = CONFIG.PreProcessing.delete_features   
    train_data = FeatureSelection(drop_column,train_data)
    test_data = FeatureSelection(drop_column,test_data)
    nom_data = FeatureSelection(drop_column,nom_data)
    train_dim = train_data.shape[1]

    ## Reshape the data for the VAE model
    x_train = np.reshape(np.array(train_data), (-1, timesteps, train_dim))
    x_test = np.reshape(np.array(test_data), (-1, timesteps, train_dim))
    nom = np.reshape(np.array(nom_data), (-1, timesteps, train_dim))

    return x_train, x_test, nom, train_dim

x_train, x_test, nom, train_dim = Preprocessing(train_data, test_data, nom_data)
print('Data preparation for training and testing is done')

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras

## For Plots font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

CONFIG.settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Selected device: {}".format(CONFIG.settings.device))

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# %%
batch = CONFIG.Models.lstm_VAE.batch_size
epoch = CONFIG.Models.lstm_VAE.epoch
latent_dim = CONFIG.Models.lstm_VAE.latent_dim
lstm1 = CONFIG.Models.lstm_VAE.layers.Lstm1
lstm2 = CONFIG.Models.lstm_VAE.layers.Lstm2
act1 = CONFIG.Models.lstm_VAE.activation.act1
act2 = CONFIG.Models.lstm_VAE.activation.act2

# %%
def MAE(data, reconstruction):
    return tf.reduce_sum(
                    keras.losses.mean_absolute_error(data, reconstruction)
            )

model = Sequential()
model.add(LSTM(lstm1, activation = act1, input_shape=(timesteps, train_dim),return_sequences=True))
model.add(LSTM(lstm2, activation = act1, return_sequences=False))
model.add(Dense(latent_dim, activation = act2))
model.add(Dense(latent_dim, activation = act2))
model.add(RepeatVector(timesteps))
model.add(LSTM(lstm2, activation= act1, return_sequences=True))
model.add(LSTM(lstm1, activation= act1, return_sequences=True))
model.add(TimeDistributed(Dense(train_dim)))
model.compile(optimizer='adam', loss=MAE)
model.summary()

# %%
def scheduler(epoch, lr):
    if epoch < 75:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(x_train, x_train, epochs=150, callbacks=[callback], shuffle=False, batch_size=128, validation_data=(nom, nom)).history



# %%
def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Training Loss', linewidth=1)
    ax.plot(history['val_loss'], 'red', label='Validation Loss', linewidth=1)
    ax.set_title('Training Loss over epochs')
    ax.set_ylabel('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    ax.grid(linestyle='-.')

plot_loss_moment(history)       #history.loss

# %%
from scipy import signal

pred = model.predict(x_test)

def deshape(data):
    data = np.mean(data,axis = 1)
    #data = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
    return data

#pred = Noise(pred)

def Plot(d, pd, title):
    plt.figure(figsize=[25,5])
    plt.plot(d[:,0], label = 'Normal')
    plt.plot(pd[:,0], label = 'Reconstructed')
    plt.legend(bbox_to_anchor=(0.1, 0.8),ncol=1, loc='lower center')
    plt.title(title)
    plt.xlabel('Time (100Hz)')
    plt.show()

Plot(deshape(x_test), deshape(pred), 'Normal vs Reconstructed data of Anomalous DC')

# %%
train_data = pd.read_csv('./Dataset/train7.csv')
nom_data = pd.read_csv('./Dataset/DriveCycle4-test-100.csv')
test_data = pd.read_csv('./Dataset/DriveCycle4-test-10.csv')
x_train, x_test, nom, _ = Preprocessing(train_data, test_data, nom_data)

# %%
def Predict(data, model):
    pred = model.predict(data, verbose=0)
    mae_loss = np.mean(np.mean(np.abs(pred - data), axis=1),axis=1)
    loss = np.mean(np.abs(pred - data), axis=2)
    loss = np.reshape(loss,(loss.shape[0]*loss.shape[1]))
    #print('loss', mae_loss)
    error_mean = np.mean(mae_loss,axis=0)
    error_std = np.std(mae_loss,axis=0)
    return mae_loss, loss, error_mean, error_std

mae_loss = np.mean(np.mean(np.abs(nom - x_test), axis=1),axis=1)
nom_mae_loss, nom_loss, nom_error_mean, nom_error_std = Predict(nom, model)
test_mae_loss, test_loss, test_error_mean, test_error_std = Predict(x_test, model)
#print(f'Reconstruction error Max-threshold: {max_threshold}')

print(nom_error_mean, test_error_mean, np.mean(mae_loss))


plt.figure(figsize=[25,10])
plt.hist(test_mae_loss, bins=50, histtype='step', label = 'Anomalous',color='r')
plt.hist(nom_mae_loss, bins=50, histtype='step', label = 'Normal',color='g')
#plt.hist(mae_loss, bins=50, histtype='step', label = 'Actual')
plt.xlabel('MAE Reconstruction Loss')
plt.ylabel('Number of samples')
plt.legend()

# %%
def Anomaly_plot(data, anomaly, title):
    data = np.mean(data,axis=1)
    plt.figure(figsize=[25,5])
    plt.plot(data[:,0], label = 'EngRPM')
    plt.scatter(anomaly.index.values,anomaly.iloc[:,0],marker='*',color = 'red', label = 'Anomaly')
    plt.legend()
    plt.title(title)
    plt.show()

def JudgeAnomaly(data,loss,threshold):
    data = np.mean(data,axis=1)
    sample = pd.DataFrame(data)
    normal = sample[loss<threshold]
    abnormal = sample[loss>=threshold]
    res = np.where(loss >= threshold, 1, 0)
    return normal, abnormal, res

threshold = 0.4
pred_normal, pred_abnormal, pred_res = JudgeAnomaly(x_test, test_mae_loss, threshold)
Anomaly_plot(x_test, pred_abnormal,'Predicted Anomaly')
#print('anomaly detection: ',pred_abnormal.shape[0]*100/x_test.shape[0],'%')

#threshold = 0.5
act_normal, act_abnormal, nom_res = JudgeAnomaly(nom, mae_loss, threshold)
Anomaly_plot(nom, act_abnormal, 'Actual Anomaly')
#print('anomaly detection: ',act_abnormal.shape[0]*100/nom.shape[0],'%')

# %%
from sklearn.metrics import roc_auc_score

def Performance(pred_abnormal,pred_normal,act_abnormal,act_normal, pred, nom):
    A = np.array(pred_abnormal.index.values.astype(int))
    B = np.array(act_abnormal.index.values.astype(int))
    TP = len([i for i in B if i in A])
    FP = len([i for i in B if i not in A])

    C = np.array(pred_normal.index.values.astype(int))
    D = np.array(act_normal.index.values.astype(int))
    TN = len([i for i in D if i in C])
    FN = len([i for i in D if i not in C])

    AUC = roc_auc_score(nom, pred)
    Accuracy = (TP+TN)/(TP+FP+TN+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    return Accuracy, F1, AUC


thresholds = np.arange(0.2,0.6,0.01)
Accuracy = []
F1 = []
AUC = []
for threshold in thresholds:
    pred_normal, pred_abnormal, pred_res = JudgeAnomaly(x_test, test_mae_loss, threshold)
    act_normal, act_abnormal, nom_res = JudgeAnomaly(nom, mae_loss, threshold)
    acc, f1, auc = Performance(pred_abnormal,pred_normal,act_abnormal,act_normal, pred_res, nom_res)
    Accuracy.append(acc*100)
    F1.append(f1*100)
    AUC.append(auc*100)


plt.figure(figsize=(10,8))
plt.plot(thresholds, Accuracy,'r', label ='Accuracy')
plt.plot(thresholds,F1,'b',label = 'F1-score')
plt.plot(thresholds,AUC,'g',label = 'AUC-score')
plt.xlabel('Threshold')
plt.ylabel('Percentage')
plt.ylim(50,100)
plt.xlim(thresholds[0],thresholds[-1])
plt.title('Threshold vs Model Performance')
plt.legend()
plt.grid(linestyle='-.')
plt.show()

max(Accuracy)

# %%



