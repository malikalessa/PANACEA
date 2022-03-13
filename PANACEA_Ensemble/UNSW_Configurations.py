import Create_Model
import Global_Config as gc
import Model_Ensembling_T_A
import Preprocessing
import Voting
from Preprocessing import *
from Create_Model import *
from Model_Ensembling_T_A import *
from Voting import *
import Global_Config as gc
config_No = 1

def Individual_Models_Predictions():
    # Predicting individual Models
     testPath = gc.UNSW_Models

     members = Create_Model.load_all_models(testPath)
     # Reading the Original Dataset
     train_scaler, test_scaler, y_train, y_test = Preprocessing.read_UNSW_Dataset()

     individual_prediction = Create_Model.predict_individual_model(members, test_scaler, y_test, testPath)


def Model_Ensembling_Train_Adversarial_UNSW_Medoids():
    # concatenating the new datasets (T+A)
    testPath = gc.UNSW_Models

    train_scaler, test_scaler, y_train, y_test = Preprocessing.read_UNSW_Dataset()
    gc.n_class = len(np.unique(y_train))

    model_medoids = []
    New_XTraining = []
    New_XValidation = []
    for i in range(gc.n_models):
        X = train_scaler
        y = y_train
        print('Model_Medoids['+str(i)+'] :' , model_medoids[i])
        print('X : ', X.shape)
        print('y : ', y.shape)
        XTraining, XValidation, YTraining, YValidation = train_test_split(X, y, stratify=y
                                                                          , test_size=0.2)  # before model building
        New_XTraining.append(XTraining)
        New_XValidation.append(XValidation)
    ## Loading the Models
    members = Create_Model.load_all_models_Medoids(testPath, model_medoids)
    gc.members = members

    # Calling the Ensemble Model Function based on the T+A Datasets
    Model_Ensembling_T_A.Ensembled_Model_Prediction(New_XTraining, YTraining, New_XValidation, YValidation, test_scaler,
                                                    y_test, testPath, config_No, n_models)


def Model_Ensembling_Train_Sequential():
    testPath = gc.UNSW_Models

    train_scaler, test_scaler, y_train, y_test = Preprocessing.read_UNSW_Dataset()
    gc.n_class = len(np.unique(y_train))

    # concatenating the new datasets (T+A)
    New_XTraining = []
    New_XValidation = []

    for i in range(gc.n_models):
        X = train_scaler
        y = y_train

        print('X : ', X.shape)
        print('y : ', y.shape)
        XTraining, XValidation, YTraining, YValidation = train_test_split(X, y, stratify=y
                                                                          , test_size=0.2)  # before model building
        New_XTraining.append(XTraining)
        New_XValidation.append(XValidation)
    ## Loading the Models
    members = Create_Model.load_all_models(testPath)
    gc.members = members

    # Calling the Ensemble Model Function based on the T+A Datasets
    Model_Ensembling_T_A.Ensembled_Model_Prediction(New_XTraining, YTraining, New_XValidation, YValidation, test_scaler,
                                                    y_test,testPath, config_No, n_models)


def create_models_UNSW():
    testPath = gc.UNSW_Models

    train_scaler, test_scaler, y_train, y_test = Preprocessing.read_UNSW_Dataset()
    gc.n_class = len(np.unique(y_train))

    for i in range(gc.n_models):
        train_dataset_adv, y_train_adv = Preprocessing.read_T_A_Datasets_UNSW(train_scaler, y_train, i)
    ### Create Models
        Create_Model.Create_Ensemble_Models(train_dataset_adv, y_train_adv, test_scaler, y_test, testPath, config_No,i)
def create_ensemble_voting():
    testPath = gc.UNSW_Models

    train_scaler, test_scaler, y_train, y_test = Preprocessing.read_UNSW_Dataset()
    model_medoids =  [ ]

    members = Create_Model.load_all_models_Medoids(testPath, model_medoids)
    gc.members = members

    Voting.voting(members,test_scaler,y_test)