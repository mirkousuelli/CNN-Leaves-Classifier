# Convolutional Neural Network: Leaves Classification
Artificial Neural Networks and Deep Learning competition 2021/2022 - Politecnico di Milano.

*Authors: **Fabio Tresoldi** and **Mirko Usuelli***

## 1. Introduction
The  given  dataset  is  unbalanced  with  respect  to  theclasses  distribution,  then  we  decided  to  apply  a  StratifiedSampling  procedure  in  order  to  preserve  the  full  dataset proportions both in training, validation and testing.This  structural  characteristic  suggests  the  usage  of  theCategorical Crossentropy as loss function, whereas as metrics we kept into account accuracy and F1-score, being thelatter more meaningful for unbalanced categories.

## 2. Data Pipeline
### 2.1.  Data  Augmentation
Due  to  the  reduced  dimension  of  the  proposed  datasetcompared to the parameters complexity of the underlying architecture, Data Augmentation purpouse consists in solvingthis issue by generating at run time several new transformedimages which allow a better generalized performance of the model.
Starting  from  Traditional  Transformations  (2.1.1),  we  tried to furtherly improved the quality of classification by introducing some Advanced Transformations (2.1.2).

#### 2.1.1.  Traditional  Transformations.
Rotating,  Zooming,Flipping, Brightness, Shifting.

#### 2.1.2.  AdvancedTransformations.
CutMix,MixUp,CutOut.

However,  these  last  Advance  Transformations  did  notincreased  the  overall  performance  in  our  specific  case,  butindeed,  they  made  the  training  process  slower  and  not  asaccurate  as  without  them,  therefore  we  decided  to  discardthese techniques.

![image](/img/leaves.jpg)

### 2.2.  Pre-Processing
Since   for   the   Features   Exctractor   (section   3.1)   werelied   on   State-of-Art   architectures   through   Fine   Tuning,   the   data   pre-processing   function   adopted   is   thesame  suggested  by  the  corresponding  model  contained  in tensorflow.keras.applications.


## 3.  Convolutional  Neural  Network
### 3.1.  Features  Extractor
We  started  comparing  the  most  famous  architectures showed during the course lectures through Transfer Learning and  a  simply  GAP  +  Softmax  layers  as  baseline  classifier. Then  we  plotted  both  Categorical  Crossentropy  loss  andAccuracy  for  the  validation  and  training  set  (dashed line) within 10 epochs:

![image](/img/comparison.jpg)

As shown in the above figure, the best models were EfficientNet and ResNet. While the latter converges faster, the former isable  to  reach  a  higher  peak  of  performance  in  both  loss and  metrics  as  epochs  go  further.  Thus,  we  have  chosen EfficientNet as Features Extractor.


### 3.2. Classifier
Starting  from a  general  classifier  model,  with the  main layers  shown  during  the  course,  we  analyzed  the  performance of each of them through Transfer Learning (**see the report for details**).
Here we show the final model we identify through several testing comparisons:

![image](/img/final.jpg)

## 4. Performance
Finally,  we  generated  the  confusion  matrix  to  identifythe correctness of classification for each class with F1-score, precision and recall, on the dataset split in 60% for trainingand 20% each for validation and testing:
- **Accuracy**: 99.77%
- **Precision**: 99.78%
- **Recall**: 99.75%
- **F1-score**: 99.76%

![image](/img/confusion_matrix.jpg)

## 5. Leadboard Evaluation
- Development phase accuracy : **94.91%**
- Final phase accuracy : **94.53%**
