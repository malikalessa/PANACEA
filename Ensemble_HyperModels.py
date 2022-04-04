from __future__ import print_function
import tensorflow as tf
import numpy as np
import Global_Config as gc
import csv
my_seed = 123
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras import callbacks
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.optimizers import Adam
import time
import keras.backend as K
import os
SavedParameters = []


def NN( params):

    X = gc.New_XTraining
    YTraining = gc.YTraining
    XVal = gc.New_XValidation
    YValidation = gc.YValidation
    print(params)
    members = gc.members

    for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                layer.trainable = True
                # rename the layer name to avoid 'unique layer name issue'
            #print('counter : ',gc.counter)
                if gc.counter == 0:
                    layer._name = 'ensemble_' +str(i) +'_' +layer.name
                print(layer.name,layer.trainable )
    gc.counter += 1
    print('counter : ', gc.counter)

    ensemble_input = [model.input for model in members]
    ensemble_output = [model.output for model in members]

    # Concatenate the outputs from all models together
    merge_output = concatenate(ensemble_output)

    hidden = Dense(params['neurons1'], activation='relu', kernel_initializer='glorot_uniform')(merge_output)
    print(type(gc.n_class))
    output = Dense(gc.n_class, activation = 'softmax' ,kernel_initializer = 'glorot_uniform' )(hidden)
    model_ensemble = Model(inputs = ensemble_input, outputs = output)
    # plot Ensembled Model
    #plot_model(model, show_shapes = True, to_file = 'Ensembled_Model_final.png')
    adam = Adam(learning_rate=params['learning_rate'])
    model_ensemble.compile(loss = 'categorical_crossentropy' ,optimizer = adam, metrics = 'accuracy')

    tic = time.time()


    YTraining = np_utils.to_categorical(YTraining, len(np.unique(YTraining)))
    YValidation = np_utils.to_categorical(YValidation, len(np.unique(YValidation)))

    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                              patience=10 ,restore_best_weights=True) ]
    h = model_ensemble.fit(X, YTraining, batch_size=params['batch'], epochs=150, verbose=2, callbacks=callbacks_list
                  , shuffle= True , validation_data=(XVal, YValidation))

    toc = time.time()
    time_tot = toc - tic
    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)

    y_test = np.argmax(YValidation, axis=1)
    Y_predicted = model_ensemble.predict(XVal, verbose=0)
    Y_predicted = np.argmax(Y_predicted, axis=1)

    return model_ensemble, h, {"val_loss": score,
                      "F1_MACRO": f1_score(y_test, Y_predicted, average='macro'),
                      "F1_MICRO": f1_score(y_test, Y_predicted, average='micro'),
                      "F1_WEIGHTED": f1_score(y_test, Y_predicted, average='weighted'),
                      "time": time_tot
                      }

def fit_and_score(params):

    model_ensemble_1, h, val = NN( params)
    print("start predict")

    test_ensemble = [gc.test_X for _ in range(len(model_ensemble_1.input))]

    Y_predicted = model_ensemble_1.predict(test_ensemble)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    elapsed_time = val['time']

    K.clear_session()

    gc.SavedParameters.append(val)

    gc.SavedParameters[-1].update({"learning_rate": params["learning_rate"], "batch": params["batch"],
                                       "neurons_layer1": params["neurons1"],
                                       "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})

    gc.SavedParameters[-1].update({
        "F1_MACRO_test": f1_score(gc.test_Y, Y_predicted, average='macro'),
        "F1_MICRO_test": f1_score(gc.test_Y, Y_predicted, average='micro'),
        "F1_WEIGHTED_test": f1_score(gc.test_Y, Y_predicted, average='weighted')})
    # Save model
    print('best score first : ', gc.best_score)

    if gc.SavedParameters[-1]["val_loss"] < gc.best_score:
        saved_model = 1
        print("new saved model:" + str(gc.SavedParameters[-1]))
        gc.best_model_ensemble = model_ensemble_1
        gc.ensemble_neurons1 = params['neurons1']
        gc.ensemble_learning_rate = params['learning_rate']
        gc.ensemble_batch = params['batch']
        gc.best_model = model_ensemble_1.get_weights()

        test_ensemble = [gc.test_X for _ in range(len(gc.best_model_ensemble.input))]
        Y_predicted = gc.best_model_ensemble.predict(test_ensemble)
        Y_predicted = np.argmax(Y_predicted, axis=1)

        gc.best_score = gc.SavedParameters[-1]["val_loss"]
        print(gc.best_model_ensemble,'validation best score : ', gc.best_score)


    SavedParameters = sorted(gc.SavedParameters, key=lambda i: i['val_loss'])

    try:
        with open(gc.test_path +'Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': val["val_loss"], 'status': STATUS_OK}


def reset_global_variables(New_XTraining,YTraining,New_XValidation,Y_Validation,test_X, test_Y):

    gc.New_XTraining = New_XTraining
    gc.YTraining = YTraining
    gc.New_XValidation = New_XValidation
    gc.YValidation  = Y_Validation
    gc.test_X = test_X
    gc.test_Y = test_Y

    gc.best_score = np.inf
    gc.best_scoreTest = 0
    gc.best_accuracy = 0
    gc.best_f1_macro = 0
    gc.best_f1_micro = 0
    gc.best_f1_weighted = 0
    gc.best_model = None
    gc.best_model_ensemble = None
    gc.best_model_test = None
    gc.best_model_f1_micro = None
    gc.best_model_f1_macro = None
    gc.best_model_f1_weighted = None
    gc.best_time = 0
    gc.SavedParameters = []

    gc.ensemble_batch = 0
    gc.ensemble_neurons1 = 0
    gc.ensemble_learning_rate = 0

def hypersearch( New_XTraining, YTraining, New_XValidation, YValidation, test_scaler, y_test,testPath,config_No):

    reset_global_variables(New_XTraining, YTraining, New_XValidation, YValidation, test_scaler, y_test)
    testPath = testPath +'ConfigNo_members' + str(config_No)
    try:
        os.remove(testPath + 'Results.csv')
    except IOError:
        print('The file has been deleted')

    gc.test_path = testPath
    print('test path : ', gc.test_path)

    space = {"batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                 "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                 "neurons1": hp.choice("neurons1", [32, 64,100, 128, 256, 512, 1024])
                          }

    trials = Trials()
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=20, trials=trials)
    # rstate= np.random.RandomState(my_seed))
    gc.best_model_ensemble.set_weights(gc.best_model)


    test_ensemble = [gc.test_X for _ in range(len(gc.best_model_ensemble.input))]


    return gc.best_model_ensemble, gc.best_time, gc.best_score

