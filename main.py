import sys
import configparser
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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



    if  (int(configuration.get('TRAIN_BASELINE'))):

       #### Creating Configurations 2,4 and 6
        import BaselineConfiguration as Baseline
        execution=Baseline.BaselineConfiguration(dsConf,configuration)
        execution.train_baseline(x_train,x_test,y_train,y_test, dsConf.get('n_class'))


    if (int(configuration.get('CREATE_ADVERSARIAL_SET'))):

        import Adversarial_Datasets as Adv
        execution = Adv.Adversarial_Datasets(dsConf, configuration)
        execution.add_Adversarial_Samples(x_train, y_train)

    if (int(configuration.get('CREATE_CANDIDATE_MODELS'))):
        ## Creating meta-models
        import CreateModels as tm
        execution=tm.CreateModels(dsConf,configuration)
        execution.createAllModels(x_train,x_test,y_train,y_test, dsConf.get('n_class'))

    if (int(configuration.get('USE_MEDOIDS'))):

        import DalexDatasets as dalex
        execution = dalex.DalexDatasets(dsConf,configuration)

        if (int(configuration.get('CREATE_DALEX_DATASET'))):

            execution.createDalexDataset(x_train,y_train)

        if (int(configuration.get('ENSEMBLE'))):

            if (int(configuration.get('TRAIN_ENSEMBLE'))):

               import Ensemble_Configuration as ensemble
               execution = ensemble.Ensemble_Configuration(dsConf,configuration)
               execution.Model_Ensembling_Train_Medoids(x_train,x_test,y_train,y_test)
            else :
                import Ensemble_Configuration as ensemble
                execution = ensemble.Ensemble_Configuration(dsConf,configuration)
                execution.create_ensemble_voting_medoids(x_train,x_test,y_train,y_test)

    if (int(configuration.get('SEQUENTIAL'))):

        if int(configuration.get('USE_SEQUENTIAL')):

            import Ensemble_Configuration as ensemble
            execution = ensemble.Ensemble_Configuration(dsConf, configuration)
            execution.Model_Ensembling_Train_Sequential(x_train,x_test,y_train,y_test)

        else:
             # Use Majority Voting with Sequential Models
            import Ensemble_Configuration as ensemble
            execution = ensemble.Ensemble_Configuration(dsConf, configuration)
            execution.create_ensemble_voting_Sequential(x_train, x_test, y_train, y_test)

    if (int(configuration.get('INDIVIDUAL_PREDICTION'))):

        import CreateModels as cr
        execution = cr.CreateModels(dsConf,configuration)
        execution.Individual_Models_Predictions(x_test,y_test)


if __name__ == "__main__":
    main()