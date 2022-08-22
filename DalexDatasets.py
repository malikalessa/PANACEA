import pandas as pd
import os
import dalex as dx
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn_extra.cluster import KMedoids
from yellowbrick.cluster import KElbowVisualizer

import Adversarial_Datasets
from Adversarial_Datasets import *

class DalexDatasets():
    def __init__(self, dsConfig,config):
        self.config = config
        self.ds = dsConfig

    def createDalexDataset(self, x_train,y_train):
        # To compute the feature relevance in the models
        relevant_vector = []

        for i in range(int(self.config.get('NUMBER_OF_MODELS'))):
            model = load_model(self.ds.get('pathModels') + 'model_'+ str(i) + '.h5')

            dataset, y_dataset = Adversarial_Datasets.read_T_A_Datasets(self,x_train,y_train,i)
            dataset.reset_index(drop=True, inplace=True)
            y_dataset.reset_index(drop=True, inplace=True)
            print('shape ', dataset.shape)
            print('shape', y_dataset.shape)
            print('y_train : ', y_dataset.value_counts())
            print('Counter : ', i)
            explainer = dx.Explainer(model, dataset, y_dataset)
            explanation = explainer.model_parts(random_state=42)
            variable_importance = pd.DataFrame(explanation.result)
            variable_importance = variable_importance.sort_values(by=['variable'], ascending=True)

            variable_importance.drop(['label'], axis=1, inplace=True)

            relevant_vector.append(variable_importance)

            ######### Computing the Feature Relevance for 100 different Models  #########################3

            df = pd.DataFrame(relevant_vector[0].set_index(['variable']), columns=['variable', 'dropout_loss'])
            df = df.rename(columns={'dropout_loss': '0'})
            df.drop(['variable'], axis=1, inplace=True)

            df = df.transpose()

        for i in range(len(relevant_vector)):
            if i == 0:
                continue
            else:
                    df1 = pd.DataFrame(relevant_vector[i].set_index(['variable']), columns=['variable', 'dropout_loss'])
                    df1.drop(['variable'], axis=1, inplace=True)
                    df1 = df1.rename(columns={'dropout_loss': str(i)})
                    df1 = df1.transpose()
                    df = df.append(df1)

        df.drop(['_full_model_'], axis=1, inplace=True)
        df.drop(['_baseline_'], axis=1, inplace=True)
        print("Dalex File Shape  : ", df.shape)
        dataset_path = self.ds.get('pathDalexDataset') + self.ds.get('Dataset_name') + '_Dalex.csv'
        print(dataset_path)
        df.to_csv(path_or_buf=dataset_path, index=False)


    def loadDalexDatasets(self):
        path = self.ds.get('pathDalexDataset') + self.ds.get('Dataset_name') + '_Dalex.csv'
        df = pd.read_csv(path)
        print('The Dalex File has been uploaded')
        n_models = int(self.config.get('NUMBER_OF_MODELS'))
        return df[:n_models]

    def Clustering_Medoids(self):

        from os.path import exists

        file_exists = exists(self.ds.get('pathDalexDataset') + self.ds.get('Dataset_name') + '_Dalex.csv')

        if (file_exists):
            print('The Dalex file exists in the specified path, it will be uploaded')
            df = DalexDatasets.loadDalexDatasets(self)
        else :
            print(' You must create Dalex File ...')
            exit(1)

        kmedoids = KMedoids(init='k-medoids++', method='pam', random_state=42)
        print(df.head())

        visualizer = KElbowVisualizer(kmedoids, k=(int(self.config.get('NUMBER_OF_MODELS'))), metric='distortion')
        visualizer.fit(df)  # Fit the data to the visualizer
        #visualizer.show()
        visualizer.show(outpath=self.ds.get('pathDalexDataset') + self.ds.get('Dataset_name') +"kelbow_minibatchkmeans.png")

        medoid_value = visualizer.elbow_value_
      
        print(medoid_value)
        kmedoids = KMedoids(n_clusters=medoid_value, init='k-medoids++', method='pam', random_state=42)
        kmedoid = kmedoids.fit(df)

        medoid = kmedoid.medoid_indices_
        medoid_list = medoid.sort()
        medoid_list = medoid.tolist()

        print(medoid_list)
        print(len(medoid))
        with open(self.ds.get('pathDalexDataset') + self.ds.get('Dataset_name') +'medoids.txt', 'w') as f:
                 f.write("medoids %s" %str(medoid_list))

        return medoid_list





