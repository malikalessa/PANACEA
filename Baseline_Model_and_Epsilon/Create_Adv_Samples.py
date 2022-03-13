import tensorflow as tf
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from Configurations import *
import Global_Config as gc

from sklearn.utils import shuffle



def Adv_Samples(model,train_scaler, y_train, test_scaler, y_test, path, config_No):

    ####### Create FastGradientMethod ##########
    train_scaler = np.asarray(train_scaler)

    classifier = TensorFlowV2Classifier(model, nb_classes=gc.n_class,input_shape=(1,train_scaler.shape[1]),
                                        loss_object = tf.keras.losses.CategoricalCrossentropy())

    eps = gc.epsilon
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    adversarial_samples = attack.generate(x=train_scaler)

    adversarial_original_samples = np.append(train_scaler, adversarial_samples, axis=0)
    y_label_adversarial = np.append(y_train, y_train, axis=0)

    adversarial_original_samples, y_label_adversarial = shuffle(adversarial_original_samples,y_label_adversarial,
                                                                 random_state=0)



    return adversarial_original_samples, y_label_adversarial, adversarial_samples
