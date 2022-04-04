import sys
import configparser
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='-1'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time


def readDataset(dsConfiguration):
    path=dsConfiguration.get('pathDataset')
    train=pd.read_csv(path+dsConfiguration.get('nameTrain'))
    test=pd.read_csv(path + dsConfiguration.get('nametest'))
    cls=dsConfiguration.get('label')
    y_train = train[cls]
    y_test = test[cls]


    try:
        train.drop([cls], axis=1, inplace=True)
        test.drop([cls], axis=1, inplace=True)
    except IOError:
        print('IOERROR')

    print(train.shape)

    return train,test, y_train,y_test


def main():

    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('PANACEA.conf')

    #this contains path dataset and models
    dsConf = config[dataset]
    #this contain the variable related to the flow of PANACEA
    configuration = config['setting']
    pd.set_option('display.expand_frame_repr', False)


    x_train,x_test, y_train,y_test=readDataset(dsConf)
    timeFile = open(dsConf.get('dataset_name')+"Time.txt","a")



    if  (int(configuration.get('TRAIN_BASELINE'))):

       #### Creating Configurations 2,4 and 6
        import BaselineConfiguration as Baseline
        execution=Baseline.BaselineConfiguration(dsConf,configuration)
        start=time.time()
        execution.train_baseline(x_train,x_test,y_train,y_test, dsConf.get('n_class'))
        totTime=time.time()-start
        print(totTime)
        timeFile.write('Time to create baseline:' )
        timeFile.write(str(totTime))
        timeFile.write('\n')


    if (int(configuration.get('CREATE_ADVERSARIAL_SET'))):
        # To creat adversarial samples datasets
        import Adversarial_Datasets as Adv
        start=time.time()
        execution = Adv.Adversarial_Datasets(dsConf, configuration)
        execution.add_Adversarial_Samples(x_train, y_train)
        totTime=time.time()-start
        print(totTime)
        timeFile.write('Time to create all 100 adversarial dataset:' )
        timeFile.write(str(totTime))
        timeFile.write('\n')
       

    if (int(configuration.get('CREATE_CANDIDATE_MODELS'))):
        ## Creating meta-models
        import CreateModels as tm
        execution=tm.CreateModels(dsConf,configuration)
        start=time.time()
        timeMax= execution.createAllModels(x_train,x_test,y_train,y_test, dsConf.get('n_class'))
        totTime=time.time()-start
        print(totTime)
        timeFile.write('Time to create all 100 models:' )
        timeFile.write(str(totTime))
        timeFile.write('\n')
        timeFile.write('Time Max to create all 100 models:' )
        timeFile.write(str(timeMax))
        timeFile.write('\n')       


    if (int(configuration.get('USE_MEDOIDS'))):
        # To use the Medoids
        import DalexDatasets as dalex    
        execution = dalex.DalexDatasets(dsConf,configuration)

        if (int(configuration.get('CREATE_DALEX_DATASET'))):
            # To create Dalex Dataset
            start=time.time()
            execution.createDalexDataset(x_train,y_train)
            totTime=time.time()-start
            timeFile.write('Time to create dalex dataset:' )
            timeFile.write(str(totTime))
            timeFile.write('\n')
        start=time.time()
        model_medoids = execution.Clustering_Medoids()
        totTime=time.time()-start
        timeFile.write('Time to create clustering:' )
        timeFile.write(str(totTime))
        timeFile.write('\n')


        if (int(configuration.get('ENSEMBLE'))):
            # To train ensemble models
            if (int(configuration.get('TRAIN_ENSEMBLE'))):
               # To train ensemble models based on XAI and Clustering
               import Ensemble_Configuration as ensemble
               execution = ensemble.Ensemble_Configuration(dsConf,configuration)
               start=time.time()
               model=execution.Model_Ensembling_Train_Medoids(x_train,x_test,y_train,y_test, model_medoids)
               totTime=time.time()-start
               timeFile.write('Time to train ensemble:' )
               timeFile.write(str(totTime))
               timeFile.write('\n')
            else :
                # To load the Emsembled models and Predict them again
                import Ensemble_Configuration as ensemble
                execution = ensemble.Ensemble_Configuration(dsConf,configuration)
                model=execution.Model_Ensembling_load_models('ensembled_model_medoids.h5')
            start = time.time()
            execution.prediction(x_test,y_test,model,'Ensembled_Model_Medoids.txt')
            totTime = time.time() - start
            timeFile.write('Time to prediction medoids:')
            timeFile.write(str(totTime))
            timeFile.write('\n')

    else:

        if int(configuration.get('TRAIN_SEQUENTIAL')):
             # To Train Sequential Models
            import Ensemble_Configuration as ensemble
            execution = ensemble.Ensemble_Configuration(dsConf, configuration)
            start=time.time()
            model=execution.Model_Ensembling_Train_Sequential(x_train,x_test,y_train,y_test)
            totTime=time.time()-start
            timeFile.write('Time to train sequential:' )
            timeFile.write(str(totTime))
            timeFile.write('\n')

        else:
             # To load the Emsembled models and Predict them again
            import Ensemble_Configuration as ensemble
            execution = ensemble.Ensemble_Configuration(dsConf, configuration)
            model=execution.model=execution.Model_Ensembling_load_models('ensembled_model_sequential.h5')
        start = time.time()
        execution.prediction(x_test, y_test, model, 'Ensembled_Model_Sequential.txt')
        totTime = time.time() - start
        timeFile.write('Time to prediction sequential:')
        timeFile.write(str(totTime))
        timeFile.write('\n')

    if (int(configuration.get('Majority_Voting'))):
        import Ensemble_Configuration as ensemble
        if (int(configuration.get('Medoid_Voting'))):
                    execution = ensemble.Ensemble_Configuration(dsConf, configuration)
                    execution.create_ensemble_voting_medoids(x_train,x_test,y_train,y_test)
        else :
            execution = ensemble.Ensemble_Configuration(dsConf,configuration)
            execution.create_ensemble_voting_Sequential(x_train,x_test,y_train,y_test)

    if (int(configuration.get('INDIVIDUAL_PREDICTION'))):

        import CreateModels as cr
        execution = cr.CreateModels(dsConf,configuration)
        start=time.time()
        execution.Individual_Models_Predictions(x_test,y_test)
        totTime=time.time()-start
        timeFile.write('Time for prediction:' )
        timeFile.write(str(totTime))
    timeFile.close()



if __name__ == "__main__":
    main()
