# i5hmCVec

In this work, we propose a computational method for identifying the 5hmC modified regions using machine learning algorithms. We applied a sequence feature embedding method based on the dna2vec algorithm to represent the RNA sequence. The ```'data'``` folder stores the data needed in the experiment. The ```'src'``` folder contains the code used in the method SGII. The ```'exp_on_weakRMdata'```folder stores the program applied for comparing i5hmCVec with WeakRM. The ```'exp_on_iRNA5hmCdata'```folder stores the program applied for comparing i5hmCVec with iRNA5hmC.

# Run

To run our supplied program, you need to configure the python 3.8 environment.

Run our code with the following command:

+ Generate the feature vector 

```
python ./src/ger_feature.py 
```

+ Generate the performace on the SVM

```
python ./src/train_svm.py --K --c --g 
```

```
--K: the k-mer embeddings used for assemblying feature vector
--c: the cost parameter in SVM
--g: the parameter in the RBF kernel function
```

+ Generate the performace on the CNN

```
python ./src/train_svm.py --K --lr --epoch 
```

```
--K: the k-mer embeddings used for assemblying feature vector
--lr: the learning rate of the SGD optimizer
--epoch: the number of epochs to train the model
```

- Generate the performace on the C4.5

  C4.5 is implemented by weka.

  ```
  javac weka_c45/java/train_c45.java
  java weka_c45/java/train_c45 k1, k2, ...,kn, C
  ```

  ```
  ki: the k-mer embeddings used for assemblying feature vector
  C: the confidence threshold for pruning
  ```
# Supplement file
This file contains the supplemental file in i5hmCVec paper
  
7z and gzip can be used as tools for unzip file.
