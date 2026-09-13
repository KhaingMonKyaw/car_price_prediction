from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from time import time
import mlflow
import pandas as pd

cars = pd.read_csv("dataset/Cars.csv")
print("Features: ", cars.columns)

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#Class
class LinearRegression(object):
    
    kfold = KFold(n_splits=5)

    #init
    def __init__(self, regularization, lr=0.001, method='stochastic', num_epochs=500, bs = 50, cv=kfold , initialization='xavier', use_momentum=True, momentum=0.9, prev_step=0):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.bs         = bs
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.initialization = initialization
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.prev_step = prev_step
        
    #mse
    def mse(self, ytrue, ypred):
        #ytrue shape:  (m, )  ==> m = number of samples
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    #fit
    def fit(self, X_train, y_train):

        # Convert pandas DataFrame/Series to NumPy arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        #create a list for keeping kfold scores
        self.kfold_scores = list()
        
        #variable to know our loss is not improving anymore
        #if the new loss !< old loss,  we stop!  (0.01 -> tolerance - tol)
        self.val_loss_old = np.inf

        #cross validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            n  = X_cross_train.shape[1]
            if(self.initialization == 'zeros'):
                self.theta = np.zeros(n)
            elif(self.initialization == 'xavier'):
                 lower = -(1.0 / np.sqrt(n))
                 upper = 1.0 / np.sqrt(n)
                 self.theta = np.random.uniform(lower, upper, n)
            else:
                raise ValueError("Invalid initialization method. Choose 'zeros' or 'xavier'.")

            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)

                for epoch in range(self.num_epochs):
                    
                    #Shuffle the data a little so that order does not impact our model
                    perm = np.random.permutation(X_cross_train.shape[0]) #perm = [2 50 67 1 .... ]
                    
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]

                    if self.method == 'stochastic':
                        for i in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[i:i+1, :]
                            y_method_train = y_cross_train[i:i+1]

                            train_loss = self._train(
                                X_method_train,
                                y_method_train
                            )
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.bs):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.bs, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.bs]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)

                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
                    
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
        
    
    #train
    def _train(self, X, y):
        #X shape: (m, n)
        #y shape: (m, )
        #theta shape: (n, )
        
        #1. predict
        yhat = self.predict(X)

        #2. grad
        m = X.shape[0]
        grad = (1/m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta)
        
        # (n, m) @ (m, ) - (m, ) = (m, ) ===> (n, )
                
        #3. update
        #self.theta = self.theta - self.lr * grad
        step = self.lr * grad
        if self.use_momentum:
            self.theta = self.theta- step + self.momentum * self.prev_step
            self.prev_step = step
        else:
            self.theta = self.theta - step
        
        #return
        return self.mse(y, yhat)
    
    
    #predict
    def predict(self, X):
        return X @ self.theta  #(m, n) @ (n, ) = (m, )  <===== y
    
    #get theta
    def _coef(self):
        return self.theta[1:]
    
    #get bias
    def _bias(self):
        return self.theta[0]

    #compute r2 score 
    def calculate_r2_score(self, ytrue, ypred):
        return 1 - (np.sum((ytrue - ypred) ** 2) / np.sum((ytrue - np.mean(ytrue)) ** 2))

    #plot feature importance
    def feature_importance(self, feature_names):
        self.coefficient = self._coef()
        #impotance = np.abs(coefficient)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, self.coefficient)
        plt.xlabel("Coefficient")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.axvline(0)
        plt.show()


#Lasso
class Lasso:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
    
    def derivation(self, theta):
        return self.l * np.sign(theta)
		
#Ridge
class Ridge:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
    
    def derivation(self, theta):
        return self.l * 2 * theta
		
#Elastic
class Elastic:
    def __init__(self, l, l_ratio):
        self.l = l
        self.l_ratio = l_ratio
        
    def __call__(self, theta): #__call__ allows us to call class as method
        l1 = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2 = (1 - self.l_ratio) * self.l * np.sum(np.square(theta)) 
        return (l1 + l2)
    
    def derivation(self, theta):
        l1 = self.l * self.l_ratio * np.sign(theta)
        l2 = 2 * self.l * (1 - self.l_ratio) * theta
        return (l1 + l2)


class NoRegularization:
    def __call__(self, theta):
        return 0.0

    def derivation(self, theta):
        return np.zeros_like(theta)

#added - km
class PolynomialRegression(LinearRegression):
    def __init__(
        self,
        degree=2,
        numeric_start=0,
        lr=0.001,
        method="mini",
        num_epochs=100,
        initialization="zeros",
        use_momentum=False
    ):
        super().__init__(
            NoRegularization(),
            lr=lr,
            method=method,
            num_epochs=num_epochs,
            initialization=initialization,
            use_momentum=use_momentum
        )
        self.degree = degree
        self.numeric_start = numeric_start
        self.polynomial_features = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )
        self.polynomial_scaler = StandardScaler()

    def _transform_features(self, X, fit=False):
        X = np.asarray(X)
        categorical_features = X[:, :self.numeric_start]
        numeric_features = X[:, self.numeric_start:]
        if fit:
            numeric_features = self.polynomial_features.fit_transform(numeric_features)
            numeric_features = self.polynomial_scaler.fit_transform(numeric_features)
        else:
            numeric_features = self.polynomial_features.transform(numeric_features)
            numeric_features = self.polynomial_scaler.transform(numeric_features)
        return np.column_stack([
            np.ones(X.shape[0]),
            categorical_features,
            numeric_features
        ])

    def fit(self, X_train, y_train):
        transformed_features = self._transform_features(X_train, fit=True)
        self.expanded_feature_count = transformed_features.shape[1]
        return super().fit(transformed_features, y_train)

    def predict(self, X):
        X = np.asarray(X)
        if X.shape[1] == self.expanded_feature_count:
            return LinearRegression.predict(self, X)
        return LinearRegression.predict(self, self._transform_features(X))
# end added - km

#inherits LinearRegression and has separate classes for each of this regularization algorithm

class LassoRegression(LinearRegression):
    def __init__(self, method, lr, l, num_epochs=500, initialization="xavier", use_momentum=True):
        self.regularization = Lasso(l)
        super().__init__(
            self.regularization,
            lr=lr,
            method=method,
            initialization=initialization,
            use_momentum=use_momentum
        )
		
class RidgeRegression(LinearRegression):
    def __init__(self, method, lr, l, initialization="xavier", use_momentum=True):
        self.regularization = Ridge(l)
        super().__init__(
            self.regularization,
            lr=lr,
            method=method,
            initialization=initialization,
            use_momentum=use_momentum
        )

class ElasticRegression(LinearRegression):
    def __init__(self, method, lr, l, l_ratio=0.5):
        self.regularization = Elastic(l, l_ratio)
        super().__init__(self.regularization, lr, method, initialization="xavier", use_momentum=True)