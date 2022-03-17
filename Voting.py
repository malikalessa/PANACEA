import numpy as np
from report import *
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import os
import report

def voting(members, test_scaler, y_test,testPath):

    predictions = [model.predict(test_scaler) for model in members]
    predictions = np.array(predictions)
    # sum across ensemble members
    summed = np.sum(predictions, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    accuracy = accuracy_score(y_test, result)
    print('acc',accuracy)

    Accuracy = accuracy_score(y_test, result)
    Confusion_matrix = confusion_matrix(y_test, result)
    Classification_report = classification_report(y_test, result)

    print('Accuracy : ', Accuracy)
    print('Confusion Matrix : \n', Confusion_matrix)
    print('Classification Report : \n', Classification_report)
    # print(model.summary())
    report_name = testPath + 'Ensembled_Voting.txt'

    try:
        os.remove(report_name)
    except:
        print('The File has been removed')
    name = 'Voting Model'
    report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
