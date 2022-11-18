# Convolutional Neural Network: Leaves Classification
Artificial Neural Networks and Deep Learning competition 2021/2022 - Politecnico di Milano.

*Authors: **Fabio Tresoldi** and **Mirko Usuelli***

## 1. Introduction
The given dataset is unbalanced concerning the class distribution,  then we decided to apply a  Stratified Sampling procedure to preserve the full dataset proportions both in training, validation, and testing. This structural characteristic suggests the usage of the Categorical Crossentropy as a loss function, whereas as metrics we kept into account accuracy and F1-score, being the latter more meaningful for unbalanced categories.

## 2. Data Pipeline
### 2.1.  Data  Augmentation
Due to the reduced dimension of the proposed dataset compared to the parameter complexity of the underlying architecture, the Data Augmentation purpose consists of solving these issues by generating at a run time several new transformed images which allow a better-generalized performance of the model.
Starting from  Traditional  Transformations  (2.1.1),  we tried to furtherly improve the quality of classification by introducing some Advanced Transformations (2.1.2).

#### 2.1.1.  Traditional  Transformations.
Rotating, Zooming, Flipping, Brightness, Shifting.

#### 2.1.2.  AdvancedTransformations.
CutMix, MixUp, CutOut.

However,  these last  Advance  Transformations did not increase the overall performance in our specific case, indeed,  they made the training process slower and not as accurate as without them,  therefore we decided to discard these techniques.

![image](/img/leaves.jpg)

### 2.2.  Pre-Processing
Since the Features Extractor (section   3.1) relied on State-of-Art architectures through Fine Tuning, the data pre-processing function adopted is the same as suggested by the corresponding model contained in TensorFlow.Keras.applications.


## 3.  Convolutional  Neural  Network
### 3.1.  Features  Extractor
We started comparing the most famous architectures shown during the course lectures through Transfer Learning and simply  GAP +  Softmax layers as baseline classifiers. Then we plotted both Categorical  Crossentropy loss and Accuracy for the validation and training set  (dashed line) within 10 epochs:

![image](/img/comparison.jpg)

The above figures show that the best models were EfficientNet and ResNet. While the latter converges faster, the former can reach a  higher peak of performance in both loss and metrics as epochs go further. Thus, we have chosen EfficientNet as Features Extractor.


### 3.2. Classifier
Starting from a general classifier model, with the main layers shown during the course, we analyzed the performance of each of them through Transfer Learning (**see the report for details**).
Here we show the final model we identify through several testing comparisons:

![image](/img/final.jpg)

## 4. Performance
Finally, we generated the confusion  matrix to identify the correctness of classification for each class with F1-score, precision, and recall, on the dataset split in 60% for training and 20% each for validation and testing:
- **Accuracy**: 99.77%
- **Precision**: 99.78%
- **Recall**: 99.75%
- **F1-score**: 99.76%

![image](/img/confusion_matrix.jpg)

## 5. Lead-board Evaluation
- Development phase accuracy: **94.91%**
- Final phase accuracy: **94.53%**
