## PANACEA(exPlAinability-based eNsemble Adversarial training for Cyber-thrEAt detection),



### The repository contains code refered to the work:
Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba

### A XAI-based approach to maximize diversity in ensembles learned with adversarial training for cyber-defense



### Code Requirements

 * Python 3.9
 * Keras 2.7
 * Tensorflow 2.7
 * Scikit learn
 * Matplotlib 3.5
 * Pandas 1.3.5
 * Numpy 1.19.3
 * Dalex 1.4.1
 * adversarial-robustness-toolbox 1.9
 * scikit-learn-extra 0.2.0
 * Hyperopt 0.2.5


### Dataset

Four different types of datasets are used in this work, NSL-KDD, UNSW-NB15, CICICD, and CIC-Maldroid20. The datasets are processed using one-hot encoder
in order to change the categorical features to numerical features.

### How to use

In the repository there are :
* The First file called baseline which is used to create the baseline model.
* The Second file is used for :
      *  Create new datasets (the datasets that contain adversarial samples).
      *  Create Baseline models.
      *  Create ensemble models.
 The Third file contain an implementation to select the enesemble models
 
 
Also, all the models that have been used in this research exist in a file called models.

