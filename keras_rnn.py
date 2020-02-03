import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from utils import DotDict
from get_dataset import get_time_data
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model

np.random.seed(2017)

def split_data(dataset='bike',sequence_length=20,ratio=1.0):

    bikes = np.array(get_time_data('data', dataset))  # (45949,1)
    opt = DotDict()
    opt.nt = bikes.shape[0]
    opt.nx = bikes.shape[1] 
    opt.nd = bikes.shape[2]
    bikes = np.reshape(bikes, (opt.nt, opt.nx * opt.nd))
    print ("Data loaded from csv. Formatting...")
    result = []
    for index in range(len(bikes) - sequence_length + 1):
        result.append(bikes[index: index + sequence_length])
    result = np.array(result)  # shape (45929, 20, 1)
    result_mean = result.mean()
    result -= result_mean
    print("Shift: ", result_mean)
    print ("Data: ", result.shape)
    # nt_train = int(round(0.95 * result.shape[0]))
    nt_train = result.shape[0] - 1
    train = result[:nt_train]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[nt_train:, :-1] 
    y_test = result[nt_train:, -1]
    # print(X_test.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], opt.nx*opt.nd))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], opt.nx*opt.nd))
    return opt, (X_train, y_train, X_test, y_test, result_mean)


def build_model(opt):
    model = Sequential()
    layers = [opt.nd*opt.nx, 50, 100, opt.nd*opt.nx]

    model.add(GRU(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae', 'mape'])
    print ("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 1
    ratio = 1
    sequence_length = 2
    dataset = 'ncov'

    if data is None:
        print ('Loading data... ')
        opt, (X_train, y_train, X_test, y_test, result_mean) = split_data(
            dataset, sequence_length, ratio)
    else:
        X_train, y_train, X_test, y_test = data

    print ('\nData Loaded. Compiling...\n')

    model = None
    if model is None:
        model = build_model(opt)
    try:
        model.fit(
            X_train, y_train,
            batch_size=512, epochs=epochs, validation_split=0.05)
        # ! save model
        model.save('keras_model.h5')
        predicted = model.predict(X_test)  # (2296, 1)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        # Evaluate
        scores = model.evaluate(X_test, y_test, batch_size=512)
        print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))

        # draw the figure
        y_test += result_mean
        predicted += result_mean
        nt_test = y_test.shape[0]
        y_test = np.reshape(y_test, (nt_test, opt.nx, opt.nd))
        predicted = np.reshape(predicted, (nt_test, opt.nx, opt.nd))
        print ('Training duration (s) : ', time.time() - global_start_time)
        show_plt(y_test, predicted)

    except Exception as e:
        print ('Training duration (s) : ', time.time() - global_start_time)
        print (str(e))

    return model, y_test, predicted


def show_plt(y_test, predict):
    # (nt_test, nx, nd)
    nt_test, nx, nd = y_test.shape
    # print(y_test.shape)
    for i in range(nd):
        # print(0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(list(range(nx)), y_test[0,:,i],label="Real")
        ax.legend(loc='upper left')
        print(1)

        plt.scatter(list(range(nx)), predict[0,:,i],label="Prediction")
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    # data_bike_num()
    run_network()
