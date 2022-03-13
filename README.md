## PANACEA(exPlAinability-based eNsemble Adversarial training for Cyber-thrEAt detection),



### The repository contains code refered to the work:
Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba

### A XAI-based approach to maximize diversity in ensembles learned with adversarial training for cyber-defense

![PANACEA](https://user-images.githubusercontent.com/38468857/158059984-f12b6302-9d07-49ae-8a59-aed1f796af6b.png)


### Code Requirements

 * [Python 3.9](https://www.python.org/downloads/release/python-390/)
 * [Keras 2.7](https://github.com/keras-team/keras)
 * [Tensorflow 2.7](https://www.tensorflow.org/)
 * [Scikit learn](https://scikit-learn.org/stable/)
 * [Matplotlib 3.5](https://matplotlib.org/)
 * [Pandas 1.3.5](https://pandas.pydata.org/)
 * [Numpy 1.19.3](https://numpy.org/)
 * [Dalex 1.4.1](https://github.com/ModelOriented/DALEX)
 * [adversarial-robustness-toolbox 1.9](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
 * [scikit-learn-extra 0.2.0](https://scikit-learn-extra.readthedocs.io/en/stable/)
 * [Hyperopt 0.2.5](https://pypi.org/project/hyperopt/)


###  Description for this repository

* Datasets and Models
   Four different types of datasets are used in this work, NSL-KDD, UNSW-NB15, CICICD, and CIC-Maldroid20. The datasets are processed using one-hot encoder
   in order to change the categorical features to numerical features. MinMax scaler has been used to normalize the datasets. The datasets and models that have been used 
   in work can be downloaded through [Datasets and Models](https://drive.google.com/drive/folders/1KnGoU2l1dxVQxIpq8AX9dAzTHuCl-_I5).
 
* Folders exist in this repository
      * Baseline_Ensemble: contains an implementation to create the baseline of PANACEA.
      * PANACEA_Ensemble : contains an implementation to create the datasets that contain adversarial samples,the baseline models and the ensembled models.
      * XAI_Clustering : contains an implementation to measure the global feature relevance of all the baseline models, to find the elbow, to cluster the features, and to return the         medoids.
     
   

### How to use

In the repository there are :
* The First file called baseline which is used to create the baseline model by running the main.py and choosing the dataset.
* The Second file is used for :
      *  Create new datasets (the datasets that contain adversarial samples).
      *  Create Baseline models.
      *  Create ensemble models.
   In this file, by running the main.py, you can choose to create datasets( adversarial datasets in order to augment them with the original datasets), and to create baseline models, predict the models, create ensembled models either sequentially or by using medoids and also to compute the majority voting for a specific number of models.
 
 * The Third file contain an implementation to select the enesemble models by using the XAI and clustering. The output for this file will be the medoids that will be passed to PANACEA_Ensemble in order to create the reuired ensemble model.
 

## Replicate the Experiments

To replicate the experiments of this work, the models and datasets that have been saved in [Datasets and Models](https://drive.google.com/drive/folders/1KnGoU2l1dxVQxIpq8AX9dAzTHuCl-_I5) can be used. The dataset Path in Preporcessing file must be configured, and the following variable must be changed :
  1. NSL_KDD_Original_Dataset = './' # To read Original Dataset
  2. NSL_KDD_datset = './' # To save and read Adversarial Samples
  3. NSL_KDD_Models = './'  # To save and read Baseline Models

  4. Maldroid20_Original_Dataset =  './'   # To read Original Dataset
  5. Maldroid20_Datasets = './' # To save and read Adversarial Samples
  6. Maldroid_20_Models = './'  # To save and read Baseline Models


  7. UNSW_Original_Dataset = './' # To read original Dataset
  8. UNSW_Datasets = './'  # To save and read Adversarial Samples
  9. UNSW_Models = './'  # To save and read  baseline Models


  10. CIC_IDS_Original_Dataset ='./'  # To read Original Dataset
  11. CIC_IDS_Datasets = './' # To save and read Adversarial Samples
  12. CIC_IDS_Models = './'  # To save and read baseline Models


