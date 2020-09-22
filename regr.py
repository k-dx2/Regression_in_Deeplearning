import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense



#loading the data from csv file
concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()

#splitting the data into predictors and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#normalizing the data by substracting the mean and dividing by the standard deviation.
predictors_norm = (predictors - predictors.mean()) / predictors.std()

n_cols = predictors_norm.shape[1] # number of predictors

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# build the model
model = regression_model()

# fit the model, fit the model,training and testing the model at the same time using fit method
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=1)
