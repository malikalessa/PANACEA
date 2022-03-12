

def  report(report_name, accuracy, confusion_matrix, classification_report, time, model,name):


    with open(report_name, 'a') as f:
        print('\n\n Name : ', name, file=f)
        print('\nNew Model  : \n',model.summary(print_fn=lambda x: f.write(x + '\n')),'\n\n', file=f)
        #print('\nHyperopt Model : \n', model_hyperopt.summary(print_fn=lambda x: f.write(x + '\n')), '\n\n', file=f)
        print('\n Accuracy is : ', accuracy, file=f)
        print('\n\n Confusion Matrix is : \n\n', confusion_matrix, file=f)
        print('\n\n Classification Report is  :  \n\n', classification_report, file=f)
        print('\n\n The Elapsed Time in Minutes is : ', time, file=f)


