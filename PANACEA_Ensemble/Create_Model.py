from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from Preprocessing import *
from report import *
import New_Hypermodel
from New_Hypermodel import *
from Global_Config import *
import Preprocessing

def Create_Ensemble_Models(train_dataset, y_train, test_scaler, y_test, testPath, config_No, model_no):

    model, best_time, best_score = New_Hypermodel.hypersearch(train_dataset, y_train, test_scaler, y_test, testPath,
                                                              model_no)

    filename = testPath+ 'model_'+ str(model_no)+'.h5'
    model.save(filename)

def load_all_models_Medoids(testPath, model_medoids):
        all_models = list()

        for i in range(gc.n_models):
            filename = testPath + 'model_' + str(model_medoids[i]) + '.h5'
            print(filename)
            model = load_model(filename)
            all_models.append(model)
        return all_models

def load_all_models(testPath):
        all_models = list()

        for i in range(gc.n_models):
            filename = testPath + 'model_' + str(i) + '.h5'
            print(filename)
            model = load_model(filename)
            all_models.append(model)
            # print('&gt;loaded %s' % filename)
        return all_models

def predict_individual_model(members ,test_scaler ,y_test, testPath):
    counter = 0
    for model in members:
    # The Prediction will be saved in the directory

        prediction = model.predict(test_scaler)
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
        report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
        counter += 1
