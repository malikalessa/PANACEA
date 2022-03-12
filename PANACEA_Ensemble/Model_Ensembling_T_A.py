import Ensemble_Hypermodel
from Global_Config import *
import Global_Config as gc
from report import *
from Ensemble_Hypermodel import *
import Save_Ensemble_Model
from Save_Ensemble_Model import *

def Ensembled_Model_Prediction(New_XTraining, YTraining, New_XValidation, YValidation, test_scaler, y_test,
    testPath, config_No, n_models):

    # Predicting Ensembled Model
    config_No = gc.n_models
    model, time, score = Ensemble_Hypermodel.hypersearch(New_XTraining, YTraining, New_XValidation, YValidation,
                                                         test_scaler, y_test, testPath, config_No)

    Save_Ensemble_Model.save_model()


    X = [test_scaler for _ in range(len(model.input))]
    print('len model input : ',len(model.input))
    predictions = model.predict(X)
    predictions = np.argmax(predictions, axis=1)
    Accuracy = accuracy_score(y_test, predictions)
    Confusion_matrix = confusion_matrix(y_test, predictions)
    Classification_report = classification_report(y_test, predictions)

    print('Accuracy : ', Accuracy)
    print('Confusion Matrix : \n', Confusion_matrix)
    print('Classification Report : \n', Classification_report)
    #print(model.summary())
    report_name = testPath + 'Ensembled_Model.txt'

    try:
        os.remove(report_name)
    except:
        print('The File has been removed')
    name = 'Ensembled Model'
    report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
