################################################################
#
# Machine Learning Algorithm for Juqex Dataset
#
# Author:
#              Alex Santiago-Anaya
#     Electrical Engineering, Virginia Tech
#
# Version: Beta 1.0 (Last Modified: March 9, 2021)
#
# Description: A Dense Neural Network (DNN) is applied
#    to the Juqex Dataset to model the fluid flow gradients.
#
################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plotloss(history):
    plt.clf()
    plt.plot(history.history['loss'], label = 'train loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('JuqexError.png')
    

def plotdata(xpred, yact):
    plt.clf()
    plt.plot(xpred, label = 'X prediction')
    plt.plot(yact, label = 'Y actual')
    plt.ylim([-30,30])
    plt.xlabel('Count')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('JuqexRandRow.png')

def plotscatter(xpred, yact):
    plt.clf()
    plt.scatter(xpred,yact)
    plt.ylim([-30,30])
    plt.xlim([-30,30])
    plt.xlabel('X predicted')
    plt.ylabel('Y actual')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('Scatter.png')

def plotRval(rvals):
    plt.clf()
    plt.plot(rvals, 'o')
    plt.xlabel('Variable')
    plt.ylabel('Pearson R Value')
    plt.grid(True)
    plt.savefig('Rvalues.png')


# read in data as csv
xdata = pd.read_csv(r'C:\Users\alexe\OneDrive\Desktop\College\Spring 2021\ML\JuqexML\JuqexML\juqex2_traindata_x.csv')
ydata = pd.read_csv(r'C:\Users\alexe\OneDrive\Desktop\College\Spring 2021\ML\JuqexML\JuqexML\juqex2_traindata_y.csv')

# random sampling across x and y
np.random.seed(2048)
percentile = .8
rows = np.random.binomial(1, percentile, size = len(xdata)).astype(bool)

xtrain = xdata[rows]
ytrain = ydata[rows]

xtest = xdata[~rows]
ytest = ydata[~rows]

# dataframe to numpy conversions
xtrain = xtrain.to_numpy()
ytrain = ytrain.to_numpy()
xtest  = xtest.to_numpy()
ytest  = ytest.to_numpy()

# normalization
xnorm = tf.keras.layers.experimental.preprocessing.Normalization()
xnorm.adapt(xtrain)

ynorm = tf.keras.layers.experimental.preprocessing.Normalization()
ynorm.adapt(ytrain)

# regression model
dnnmodel = tf.keras.Sequential([
    xnorm,
    tf.keras.layers.Dense(588, activation = 'softplus'),
    tf.keras.layers.Dense(980, activation = 'relu'),
    tf.keras.layers.Dense(1960, activation = 'softplus'),
    tf.keras.layers.Dense(980, activation = 'selu'),
    tf.keras.layers.Dense(588, activation= 'softplus'),
    tf.keras.layers.Dense(196) # Must be same size as output
    ])

dnnmodel.compile(optimizer = 'adam', loss = 'mae')
hist = dnnmodel.fit(xtrain, ytrain, epochs = 500, batch_size = 80)
dnnmodel.save('JuqexModel')

xpredict = dnnmodel.predict(xtest)
plotloss(hist)
plotdata(xpredict[13], ytest[13])
plotscatter(xpredict[13], ytest[13])

corr = np.zeros(196)
totalavg = 0

for i in range(196):
    corr[i], _ = pearsonr(xpredict[:,i], ytest[:,i])
    totalavg += corr[i]

totalavg = totalavg/196

print(totalavg)
plotRval(corr)

#####################################################################
# TODO:
#  1) Determine proper loss function (Complete - Verification Needed)
#  2) Adjust model parameters to minimize loss (In Progress)
#  3) Verify using test data (TBD)
#  4) Determine applications beyond test data verification (TBD)
#  5) Plot sample row comparison (Complete)
####################################################################