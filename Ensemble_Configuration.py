import Ensemble_HyperModels
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from CreateModels import *
from DalexDatasets import *
from Save_Ensemble_Model import *
import Save_Ensemble_Model
import report
import Voting
import time


class Ensemble_Configuration():

    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig



    def Model_Ensembling_Train_Medoids(self,x_train,x_test,y_train,y_test,medoids):

        # concatenating the new datasets (T+A)
        testPath =  self.ds.get('pathModels')
        n_class = int(self.ds.get('n_class'))
        gc.n_class = n_class

        model_medoids = medoids
        New_XTraining = []
        New_XValidation = []
        for i in range(len(model_medoids)):
            X = x_train
            y = y_train

            print('Model_Medoids[' + str(i) + '] :', model_medoids[i])
            print('X : ', X.shape)
            print('y : ', y.shape)
            XTraining, XValidation, YTraining, YValidation = train_test_split(X, y, stratify=y
                                                                              , test_size=0.2)  # before model building
            New_XTraining.append(XTraining)
            New_XValidation.append(XValidation)
        ## Loading the Models
        members = CreateModels.load_all_models_Medoids(self,model_medoids)
        gc.members = members
        config_No = len(model_medoids)
        # Calling the Ensemble Model Function based on the T+A Datasets
        model, time, score = Ensemble_HyperModels.hypersearch(New_XTraining, YTraining, New_XValidation, YValidation,
                                                        x_test,y_test, testPath, config_No)

        Save_Ensemble_Model.save_model(testPath)

        X = [x_test for _ in range(len(model.input))]
        print('len model input : ', len(model.input))
        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        Accuracy = accuracy_score(y_test, predictions)
        Confusion_matrix = confusion_matrix(y_test, predictions)
        Classification_report = classification_report(y_test, predictions)

        print('Accuracy : ', Accuracy)
        print('Confusion Matrix : \n', Confusion_matrix)
        print('Classification Report : \n', Classification_report)
        # print(model.summary())
        report_name = testPath + 'Ensembled_Model_Medoids.txt'

        try:
            os.remove(report_name)
        except:
            print('The File has been removed')
        name = 'Ensembled Model Medoids'

        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)


    def Model_Ensembling_load_Ensembled_Models(self,x_train,x_test,y_train,y_test):
        # This Function is used to load the saved ensembled models in order to predict them
        # concatenating the new datasets (T+A)
        testPath =  self.ds.get('pathModels')

        # To load the Ensemble Model
        model = load_model(testPath+'ensembled_model_8.h5')

        X = [x_test for _ in range(len(model.input))]
        print('len model input : ', len(model.input))
        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        Accuracy = accuracy_score(y_test, predictions)
        Confusion_matrix = confusion_matrix(y_test, predictions)
        Classification_report = classification_report(y_test, predictions)

        print('Accuracy : ', Accuracy)
        print('Confusion Matrix : \n', Confusion_matrix)
        print('Classification Report : \n', Classification_report)
        # print(model.summary())
        report_name = testPath + 'Ensembled_Model_Medoids.txt'

        try:
            os.remove(report_name)
        except:
            print('The File has been removed')
        name = 'Ensembled Model Medoids'

        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)




    def Model_Ensembling_Train_Sequential(self,x_train,x_test,y_train,y_test):
        testPath = self.ds.get('pathModels')
        n_class = self.ds.get('n_class')
        n_class = int(n_class)
        print(n_class)
        gc.n_class = n_class

        n_models = int(self.config.get('NUMBER_OF_MODELS'))

        # concatenating the new datasets (T+A)
        New_XTraining = []
        New_XValidation = []
        for i in range(n_models):
            X = x_train
            y = y_train

            print('X : ', X.shape)
            print('y : ', y.shape)
            XTraining, XValidation, YTraining, YValidation = train_test_split(X, y, stratify=y
                                                                              , test_size=0.2)  # before model building
            New_XTraining.append(XTraining)
            New_XValidation.append(XValidation)
        ## Loading the Models
        members = CreateModels.load_all_models(self)
        gc.members = members
        config_No = n_models
        # Calling the Ensemble Model Function based on the T+A Datasets
        model, time, score = Ensemble_HyperModels.hypersearch(New_XTraining, YTraining, New_XValidation, YValidation,
                                                              x_test, y_test, testPath, config_No)

        Save_Ensemble_Model.save_model(testPath)

        X = [x_test for _ in range(len(model.input))]
        print('len model input : ', len(model.input))
        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        Accuracy = accuracy_score(y_test, predictions)
        Confusion_matrix = confusion_matrix(y_test, predictions)
        Classification_report = classification_report(y_test, predictions)

        print('Accuracy : ', Accuracy)
        print('Confusion Matrix : \n', Confusion_matrix)
        print('Classification Report : \n', Classification_report)
        # print(model.summary())
        report_name = testPath + 'Ensembled_Model_Sequential.txt'

        try:
            os.remove(report_name)
        except:
            print('The File has been removed')
        name = 'Ensembled Model Sequential'
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)

    def create_ensemble_voting_medoids(self,x_train,x_test,y_train,y_test,model_medoids):


        testPath = self.ds.get('pathModels')
        n_class = int(self.ds.get('n_class'))
        gc.n_class = n_class
        n_models = self.config.get('NUMBER_OF_MODELS')
        members = CreateModels.load_all_models_Medoids(self,model_medoids)
        gc.members = members

        Voting.voting(members, x_test, y_test,testPath)

    def create_ensemble_voting_Sequential(self, x_train, x_test, y_train, y_test,):
        testPath = self.ds.get('pathModels')
        n_class = self.ds.get('n_class')
        n_class = int(n_class)
        gc.n_class = n_class
        n_models = self.config.get('NUMBER_OF_MODELS')

        members = CreateModels.load_all_models(self)
        gc.members = members
        Voting.voting(members, x_test, y_test, testPath)
