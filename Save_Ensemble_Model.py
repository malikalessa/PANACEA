from keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization
from keras.models import Model
import Global_Config as gc
from tensorflow.keras.optimizers import Adam



def save_model(testPath, name):
    members = gc.members
    print('members : ', len(members))
    counter = 0
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = True
            # rename the layer name to avoid 'unique layer name issue'
            layer._name = 'ensemble_' + str(i) + '_' + layer.name
            print(layer.name, layer.trainable)

            # To define Multiheaded input
    ensemble_input = [model.input for model in members]
    ensemble_output = [model.output for model in members]
    # Concatenate the outputs from all models together
    merge_output = concatenate(ensemble_output)
    # Adding two hidden layers for the concatenated output
    hidden = Dense(gc.ensemble_neurons1, activation='relu', kernel_initializer='glorot_uniform')(merge_output)

    output = Dense(gc.n_class, activation='softmax', kernel_initializer='glorot_uniform')(hidden)
    model = Model(inputs=ensemble_input, outputs=output)

    adam = Adam(learning_rate=gc.ensemble_learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    #print('len model2 layers ',len(model.layers))
    model.set_weights(gc.best_model)
    model.save(testPath + name)
    #model.save(testPath + 'ensembled_model_'+str(len(members))+'.h5')

