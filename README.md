# Predicting Length of Stay with Boosting Algorithm

## Gradient Boost Algorithm 
Ensemble learning or boosting has become one of the most promising approaches for analysing data in machine learning domain. The method was initially proposed as ensemble methods based on the principle of generating multiple predictions and average voting among individual classifiers.

### AdaBoost   
AdaBoost or Adaptive Boosting is the Boosting ensemble model,a statistical classification meta-algorithm refers to a particular method of training a boosted classifier. The method automatically adjusts its parameters to the data based on the actual performance in the current iteration.

### CATBoost
CATBoost - Provides a gradient boosting framework which among other features attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. Catboost  calculates the residual of each data point and uses the model trained with other data. In this way, each data point gets different residual data. These data are evaluated as targets, and the training times of the general model are as many as the number of iterations. Since many models will be implemented by definition, this computational complexity seems very expensive and takes too much time.
