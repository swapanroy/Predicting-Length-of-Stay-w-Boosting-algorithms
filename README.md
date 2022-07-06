# Predicting Length of Stay with Boosting Algorithm
 
**Topics covered : 

1. What is Boosting ?

2. What is a Boosting Algorithms ?

3. Examples of Boosting Algorithm **

For end to end implementation refer to : [HealthCare Predicting Length of stay with boosting algorithms](https://www.kaggle.com/code/swapanroy/predicting-length-of-stay-w-boosting-algorithms/notebook?scriptVersionId=100100872) 

Git Hub to pull the code : [Github/swapanroy](https://github.com/swapanroy/Predicting-Length-of-Stay-w-Boosting-algorithms) 


#### 1. What is Boosting ?
Boosting ("**_to Boost_**", in english meaning _help or encourage to increase or improve_.) is a method used in machine learning to improve machine models' predictive accuracy and performance.

#### 2. What is a Boosting Algorithms ?
Ensemble learning or boosting has become one of the most promising approaches in machine learning domain. The ensemble method is based on the principle of generating multiple predictions and average voting among individual classifiers.

Two implementation of Boosting Algorithm
#### AdaBoost
_AdaBoost or Adaptive Boosting_ is the Boosting ensemble model,a statistical classification meta-algorithm refers to a particular method of training a boosted classifier. The method automatically adjusts its parameters to the data based on the actual performance in the current iteration.

#### CATBoost
_CATBoost_ - Provides a gradient boosting framework which among other features attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. Catboost calculates the residual of each data point and uses the model trained with other data. In this way, each data point gets different residual data. These data are evaluated as targets, and the training times of the general model are as many as the number of iterations. Since many models will be implemented by definition, this computational complexity seems very expensive and takes too much time.




#####  Python implemenation will use Sklearn and Catboost library 


#### Data Source: [Kaggle- HealthCare data to predict length of stay](https://www.kaggle.com/code/swapanroy/predicting-length-of-stay-w-boosting-algorithms/data?scriptVersionId=100100872)


#### Key Libraries 

##### Libraries needed for data Analysis 

`import pandas as pd
import numpy as np`

##### Libraries needed for models and visualization 
`import matplotlib.pyplot as plt
import seaborn as sns`

##### Libraries needed for pre-processing 
`from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler`

##### Libraries needed for models
`from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_confusion_matrix`

##### Libraries needed for validation 
`from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split``


#### Read data and pre-process data for models 

`df_train=pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv',sep=',')

'plt.figure(figsize=(20,5))
x = sns.countplot(df_train['Stay'], order = df_train['Stay'].value_counts().index)
for i in x.containers:
 x.bar_label(i,)'

#### Merging Train and Test on multi dimention to pre-process data.
`df_merge = [df_train, df_test]
df_merge[0]`


#### Binning 
`age_value = {'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}
stay_value = {'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9, 'More than 100 Days': 10}`

#### Replacing Age and Stay with Int values
`df_merge[0]['Age'] = df_merge[0]['Age'].replace(age_value.keys(), age_value.values())
df_merge[0]['Stay'] = df_merge[0]['Stay'].replace(stay_value.keys(), stay_value.values())
df_merge[1]['Age'] = df_merge[1]['Age'].replace(age_value.keys(), age_value.values())`




![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/xpkm4zf36a2tat1u1v97.jpg)
#### Applying Adaboost classifier 
An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of error classified instances are adjusted.

```
ada_classifier = AdaBoostClassifier(n_estimators=5)
ada_classifier.fit(X_train, y_train)
pred_ada = ada_classifier.predict(X_test)

# Cross-validation
scores = cross_val_score(ada_classifier,X_test,y_test, cv=12)
print('Accuracy score',round(scores.mean() * 100,2))
print('Confusion Matrix\n',confusion_matrix(y_test, pred_ada))
```


`Accuracy score 33.8`


#### Applying CatBoost - Gradient Algorithm
Gradient boosting algorithm works by building simpler (weak) prediction models sequentially where each model tries to predict the error left over by the previous model. It find uses in search, recommendation systems, personal assistant, self-driving cars, weather prediction.


##### Important Parameters of CatBoost Model

1. iterations - It accepts integer specifying the number of trees to train. The default is 1000.
2. learning_rate - It specifies the learning rate during the training process. The default is 0.03.
3. l2_leaf_reg - It accepts float specifying coefficient of L2 regularization of a loss function. The default value is 3.
4. loss_function - It accepts string specifying metric used during training. The gradient boosting algorithm will try to minimize/maximize loss function output depending on the situation.
5. eval_metric - It accepts string specifying metric to evaluate on evaluation set given during training. It has the same options as that of loss_function.


```
model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.3,
                           depth=10,
                           l2_leaf_reg = 3,
                           random_strength =2,
                           loss_function='MultiClass',
                           eval_metric='MultiClass')
```

##### fitting model and predicting accuracy 

```
model.fit(X_train,
          y_train,
          eval_set=eval_dataset,
          verbose=True)

print(model.get_best_score())
cm = get_confusion_matrix(model, eval_dataset)
print(cm)
predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
ax = sns.heatmap(cm, linewidth=1)
plt.show()
print("catboost Acc : ", predict_accuracy_on_test_set)
```

`Accuracy : 0.40188104509483735 `



References: 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

https://github.com/catboost/catboost

Photo by [Jen Theodore](https://unsplash.com/@jentheodore) on Unsplash

Cross-posting on https://dev.to/swapanroy/boosting-algorithms-1jmb 



