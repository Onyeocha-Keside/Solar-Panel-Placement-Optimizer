import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


#Data Collection

def collect_data(csv_file):
    return pd.read_csv(csv_file)

#Preprocess
def preprocessed_data(data):

    #drop missing value
    data = data.dropna()

    #allocate target value
    X = data.drop("suitability_score", axis = 1)
    y = data["suitability_score"]

    #normalize data

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    return X_scaled, y

#model building

def training_model(x,y):

    model = Sequential([
        Dense(64, activation='relu', input_shape=(x.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    #model.fit(x, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    return model

#making predictions
def predictions(model, input_data):
    return np.squeeze(model.predict(input_data))

#store results
#def store_results(location, prediction, database):
#    print(f"Location: {location}, Predicted Suitability: {prediction:.2f}")
def store_results(location, prediction, database):
    with open('predictions.csv', 'a') as f:
        f.write(f"{location}, {prediction:.2f}\n")

#atlas
def solar_optimization_pipeline(data_file):
    data = collect_data(data_file)
    X, y = preprocessed_data(data)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #train model
    model = training_model(X_train, y_train)
    prediction = predictions(model, X_test)

    #store the result
    for i, pred in enumerate(prediction):
        store_results(f'Location_{i}', pred, None)

if __name__ == '__main__':
    solar_optimization_pipeline('solar_data.csv')