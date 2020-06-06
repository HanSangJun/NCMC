# Nearest Class Multiple Centroids (NCMC)

## Information
Modified version of T. Mensink's Matlab code  
@ version : v0.2  
@ date : Created on Tue, Dec, 11, 2018  

## References
[1] T. Mensink et al., "Distance-Based Image Classification: Generalizing to New Classes at Near Zero Cost,"
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013  

## Getting Started
Follow demo.ipynb  

## Method
```
N : the number of data, D : features, d : features in semantic space, C : the number of class 
```

### fit
Train model using stochastic gradient descent
```
fit(X, y, X_val, y_val, semantic_dim=500, learning_rate=0.001, init=None, num_centroids=3, reg=0,
            max_iters=100, batch_size=256, decay=0.8, num_decay=5, stop_tol=0.001, num_stop_tol=30):
```

Inputs :
- X : A numpy array of shape (N, D) for training
- y : A numpy array of shape (N,) for training
- X_val : For validation (if no validation set, set X_val = None)
- y_val : For validation (if no validation set, set y_val = None)
- semantic_dim : Feature dimension in semantic space (d,)
- learning_rate : Learning rate for optimization
- init : if None, normal random distribution, or weight matrix W
- num_centroids : The number of initial centroids 
- max_iters : the number of iterating
- batch_size : the number of randomly sampling data for one iteration
- decay : the ratio of decaying learning rate
- num_decay : Epoch cycle for decaying learning rate
- stop_tol : tolerance for early stopping 
- num_stop_tol : the counting of violating tolerance for early stopping

Outputs:
- history : History dictionary containing training loss and accuracy

### predict
Prediction

```
predict(X, y=None)
```

Inputs :
- X : A numpy array of shape (N, D)
- y : A numpy array of shape (N,)

Outputs:
- loss (if label is not contained, it returns 0)
- accuracy (if label is not contained, it returns softmax matrix)
- y_pred

### getDist
Get distance matrix

```
getDist(X)
```

Inputs :
- X : A numpy array of shape (N, D)

Outputs:
- dist

### predictDist
Prediction by distance  
Before implement this function, required to decide a threshold that separates unseen class data  
For example, ncmml.setThreshold(threshold)

```
predictDist(X, y=None, new_class=999)
```

Inputs :
- X : A numpy array of shape (N, D)
- y : A numpy array of shape (N,)
- new_class : assign new class to the data which is far from each centroids 

Outputs:
- accuracy (if label is not contanied, it returns 0)
- dist (if label is not contanied, it returns distance matrix)
- y_pred