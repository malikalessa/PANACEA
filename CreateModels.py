import Global_Config as gc
import Baseline_HyperModel
from Adversarial_Datasets import *
import Adversarial_Datasets as Adv
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import report


class CreateModels():
    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig

    def createAllModels(self,x_train,x_test, y_train,y_test, n_class):

        No_Models  = int(self.config.get('NUMBER_OF_MODELS'))
        #here the code to create the base candidate models

        path = self.ds.get('pathModels')
        gc.n_class = int(n_class)
        timeMax=0

        for i in range (No_Models):
            config_No = i

            Adv_train, y_adv_train = Adversarial_Datasets.read_T_A_Datasets(self,x_train,y_train,i)
            start=time.time()

            model, time1, score = Baseline_HyperModel.hypersearch(Adv_train, y_adv_train, x_test, y_test, path,
                                                                       config_No)
            time1=time.time()-start
            if time1 > timeMax:
               timeMax= time1                                                   
            filename = path + 'model_' + str(i) + '.h5'
            model.save(filename)
        return timeMax


    def load_all_models_Medoids(self,model_medoids):
            all_models = list()

            for i in range(len(model_medoids)):
                filename = self.ds.get('pathModels') + 'model_' + str(model_medoids[i]) + '.h5'
                print(filename)
                model = load_model(filename)
                all_models.append(model)
            return all_models

    def load_all_models(self):
        all_models = list()
        for i in range(int(self.config.get('NUMBER_OF_MODELS'))):
            filename = self.ds.get('pathModels') + 'model_' + str(i) + '.h5'
            print(filename)
            model = load_model(filename)
            all_models.append(model)
        return all_models

    def Individual_Models_Predictions(self, x_test, y_test):
        # Predicting individual Models
        testPath = self.ds.get('pathModels')

        members = CreateModels.load_all_models(self)
        counter = 0
        for model in members:
            # The Prediction will be saved in the directory

            prediction = model.predict(x_test)
            prediction = np.argmax(prediction, axis=1)
            Accuracy = accuracy_score(y_test, prediction)
            Confusion_matrix = confusion_matrix(y_test, prediction)
            Classification_report = classification_report(y_test, prediction)

            report_name = testPath + 'Model_Predict_ ' + str(counter) + '.txt'
            try:
                os.remove(report_name)
            except:
                print('')
            name = 'Model_ ' + str(counter)
            report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
            counter += 1
        print('Predictions have been saved in : ', self.ds.get('pathModels'))
