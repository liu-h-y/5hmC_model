# i5hmCVec

In this work, we propose a computational method for identifying the 5hmC modified regions using machine learning algorithms. We applied a sequence feature embedding method based on the dna2vec algorithm to represent the RNA sequence. 

# Setup instructions
i5hmCVec requires the enviroment with Python 3.8. The modules requried by i5hmCVec is provied in the file ```'requirements.txt'```, and you can install them by:
```bash
pip install requirements.txt
```

# Reproducing experiments
## Generating the performance of i5hmCVec on SVM
Step 1: Generate the feature vector 
```bash
python ./i5hmCVec_src/ger_feature.py 
```
Step 2: Train i5hmCVec on SVM
```bash
python ./i5hmCVec_src/train_svm.py --K --c --g 
```

```
--K: the k-mer embeddings used for assemblying feature vector
--c: the cost parameter in SVM
--g: the parameter in the RBF kernel function
```

## Generating the performance of i5hmCVec on CNN
Step 1: Generate the feature vector 
```bash
python ./i5hmCVec_src/ger_feature.py 
```
Step 2: Train i5hmCVec on CNN
```bash
python ./i5hmCVec_src/train_cnn.py --K --lr --epoch 
```

```
--K: the k-mer embeddings used for assemblying feature vector
--lr: the learning rate of the SGD optimizer
--epoch: the number of epochs to train the model
```
## Generating the performance of i5hmCVec on C4.5
We implement the i5hmCVec on C4.5 with java. The ```'train_c45'``` folder contains the program code.  
- The  ```'Data'``` folder contains the digital feature in the same data splitting with SVM and CNN.
- The ```'train_c45.java'``` scripts in ```'java'``` folder is used for Generating the performance of i5hmCVec on C4.5.
  - The variable ```'ArrayList<Integer> ks'``` represents the k-mer embeddings used for assemblying feature vector.
  - The variable ```'String C'``` represents the confidence threshold for pruning.

## Generating the performance of i5hmCVec on iRNA5hmC dataset
Step 1: Feature encoding on i5hmCVec dataset
```bash
python ./i5hmCVec_on_iRNA5hmC_data/ger_feature.py
```
Step 2: Train i5hmCVec on iRNA5hmC dataset
```bash
python ./i5hmCVec_on_iRNA5hmC_data/model.py 
```

## Generating the performance of i5hmCVec on WeakRM dataset
Step 1: Feature encoding on WeakRM dataset
```bash
python ./i5hmCVec_on_WeakRM_data/ger_feature.py
```
Step 2: Train i5hmCVec on WeakRM dataset
```bash
python ./i5hmCVec_on_WeakRM_data/model.py
```

# File descriptions
## i5hmCVec_src
This folder contains the program for generating the performance of i5hmCVec on three classifiers, including SVM, CNN, and C4.5.  
- ```'data.csv'``` contains the dataset proposed in i5hmCVec.
- ```'feature.csv'``` contains the digital features of dataset with feature encoding method proposed in this study.
- ```'ger_feature.py'``` generates the feature vector for dataset.
- ```'train_svm.py'``` generates the performance of i5hmCVec on SVM.
- ```'train_cnn.py'``` generates the performance of i5hmCVec on CNN.
- ```'train_c45'``` contains the program for generating the performance of i5hmCVec on C4.5. We implement this C4.5 project with java.

## i5hmCVec_on_iRNA5hmC_data  
This folder contains the program for generating the performance of i5hmCVec on the dataset from iRNA5hmC.
- ```'iRNA5hmC_data.txt'``` contains the dataset from iRNA5hmC.
- ```'feature.csv'``` contains the digital features of dataset from iRNA5hmC with feature encoding method proposed in this study.
- ```'ger_feature.py'``` is used to generated the feature proposed in i5hmCVec on the dataset from iRNA5hmC.
- ```'model.py'``` is used to performed 10 times 5-fold cross-validation on model.

## i5hmCVec_on_WeakRM_data
This folder contains the program for generating the performance of i5hmCVec on the dataset from WeakRM.
- ```'data'``` contains the dataset from WeakRM, including train dataset, valid dataset, test dataset.
- ```'feature'``` contains the digital features of dataset from WeakRM with feature encoding method proposed in this study.
- ```'ger_feature.py'``` is used to generated the feature proposed in i5hmCVec on the dataset from WeakRM.
- ```'model.py'``` is used to evaluate the performance of i5hmCVec on the dataset from WeakRM.

## dm3.w2v
This file contains the k-mer embeddings trained by dna2vec on mm9.
  

