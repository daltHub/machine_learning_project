"""
Contains functions that help with evaluating performance of models
"""
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

def test():
    """Test function"""
    print("Imported Correctly")

def train_lasso(input_features, target_feature, c_value):
    """
    Fits a lasso regressor to a set of data

    Parameters
    ----------
    input_features : numpy.ndarray
        features

    target_feature : numpy.ndarray
        target features

    c_value : float
        used to determine alpha training parameter

    Returns
    ----------
    model : linear_model._coordinate_descent.Lasso

    """

    model = linear_model.Lasso(alpha=1/(2*c_value), max_iter=1000000000)
    model.fit(input_features, target_feature)
    # print('coeffs =',model.coef_)
    return model

def train_lasso_for_C(input_features, target_feature, C_array):
    """
    Fits a lasso regressor to a set of data for a range of values for C

    Parameters
    ----------
    input_features : numpy.ndarray
        features

    target_feature : numpy.ndarray
        target features

    C_array : float array
        used to determine alpha training parameter

    Returns
    ----------
    model_array : array of linear_model._coordinate_descent.Lasso

    """
    model_array = np.ndarray(shape=(len(C_array)), dtype= linear_model._coordinate_descent.Lasso )
    for i in range(len(C_array)):
        model_array[i] = train_lasso(input_features, target_feature, C_array[i])
    return model_array


def generate_predictions(model_array, test_data):
    """
    Generates predictions on a dataset using an array of models 

    Parameters
    ----------
    model_array : array of linear_model._coordinate_descent.Lasso
        features

    test_data : numpy.ndarray
        input features

    Returns
    ----------
    predictions_array : array

    """
    predictions_array = np.ndarray(shape=(len(model_array) ,len(test_data)), dtype= np.float64 )
    for i in range(len(model_array)):
        predictions_array[i,:] = model_array[i].predict(test_data)

    return predictions_array

def train_Kfold_lasso(X_features, y_features, c_value):
    """
    Uses K-fold cross validation 

    Parameters
    ----------
    X_features : array 
        features

    y_features : array
        target features

    c_value : float
        parameter for training

    Returns
    ----------
    mean error : float

    standard error : float

    """
    kf = KFold(n_splits=5)
    model = linear_model.Lasso(alpha=1/(2*c_value))
    errs = []
    # model = linear_model.Lasso(alpha=1/(2*c_value), max_iter=1000000000).fit()
    for train, test in kf.split(X_features):
        model.fit(X_features[train],y_features[train])
        ypred = model.predict(X_features[test])
        from sklearn.metrics import mean_squared_error
        # print("square error %f"%(mean_squared_error(y_features[test],ypred)))
        errs.append(mean_squared_error(y_features[test],ypred))
    # print(np.mean(errs))
    return np.mean(errs), np.std(errs)

def Kfold_for_C_lasso(X_features, y_features, C_range):
    """
    Uses K-fold cross validation with varied values of C

    Parameters
    ----------
    X_features : array 
        features

    y_features : array
        target features

    C_range : array of float
        parameters for training

    Returns
    ----------
    nothing
    """
    error_array = np.zeros(len(C_range))
    std_dev_array = np.zeros(len(C_range))
    for i in range(len(C_range)):
        # print("\n\n C = %f"%(C_range[i]))
        error_array[i], std_dev_array[i] = train_Kfold_lasso(X_features, y_features, C_range[i])


    # print(error_array)

    plt.figure("K-fold Error for varied C")
    plt.errorbar(C_range, error_array, yerr=std_dev_array)
    plt.xlabel('C value')
    plt.ylabel('Mean Squared Error')
    plt.title('K-fold Error for varied C')
    # x = np.arange(len(error_array))
    # plt.bar(x, C_range, error_array)

def train_ridge(input_features, target_feature, c_value):
    """
    Fits a ridge regressor to a set of data

    Parameters
    ----------
    input_features : numpy.ndarray
        features

    target_feature : numpy.ndarray
        target features

    c_value : float
        used to determine alpha training parameter

    Returns
    ----------
    model : linear_model.Ridge

    """
    ### train a Ridge regression model.
    ### alpha seems to be approximately 1/c
    ### https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier
    model = linear_model.Ridge(alpha=1/(2*c_value), max_iter=1000000000)
    model.fit(input_features, target_feature)
    # print('coeffs =',model.coef_)
    return model


def train_ridge_for_C(input_features, target_feature, C_array):
    """
    Fits a lasso regressor to a set of data for a range of values for C

    Parameters
    ----------
    input_features : numpy.ndarray
        features

    target_feature : numpy.ndarray
        target features

    C_array : float array
        used to determine alpha training parameter

    Returns
    ----------
    model_array : array of linear_model._coordinate_descent.Lasso

    """
    model_array = np.ndarray(shape=(len(C_array)), dtype= linear_model._coordinate_descent.Lasso )
    for i in range(len(C_array)):
        model_array[i] = train_ridge(input_features, target_feature, C_array[i])
    return model_array

def train_Kfold_ridge(X_features, y_features, c_value):
    """
    Uses K-fold cross validation 

    Parameters
    ----------
    X_features : array 
        features

    y_features : array
        target features

    c_value : float
        parameter for training

    Returns
    ----------
    mean error : float

    standard error : float

    """
    kf = KFold(n_splits=5)
    model = linear_model.Ridge(alpha=1/(2*c_value))
    errs = []
    # model = linear_model.Lasso(alpha=1/(2*c_value), max_iter=1000000000).fit()
    for train, test in kf.split(X_features):
        model.fit(X_features[train],y_features[train])
        ypred = model.predict(X_features[test])
        from sklearn.metrics import mean_squared_error
        # print("square error %f"%(mean_squared_error(y_features[test],ypred)))
        errs.append(mean_squared_error(y_features[test],ypred))
    # print(np.mean(errs))
    return np.mean(errs), np.std(errs)


def Kfold_for_C_ridge(X_features, y_features, C_range):
    """
    Uses K-fold cross validation with varied values of C

    Parameters
    ----------
    X_features : array 
        features

    y_features : array
        target features

    C_range : array of float
        parameters for training

    Returns
    ----------
    nothing
    """
    error_array = np.zeros(len(C_range))
    std_dev_array = np.zeros(len(C_range))
    for i in range(len(C_range)):
        # print("\n\n C = %f"%(C_range[i]))
        error_array[i], std_dev_array[i] = train_Kfold_ridge(X_features, y_features, C_range[i])


    # print(error_array)

    plt.figure("K-fold Error for varied C")
    plt.errorbar(C_range, error_array, yerr=std_dev_array)
    plt.xlabel('C value')
    plt.ylabel('Mean Squared Error')
    plt.title('K-fold Error for varied C')
    # x = np.arange(len(error_array))
    # plt.bar(x, C_range, error_array)