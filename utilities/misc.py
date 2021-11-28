"""
Contains miscellaneous functions that help with evaluating performance of models
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def test():
    """Test function"""
    print("Imported Correctly")

def plot_training_data(input_features, target_feature, dataset_id=None, x_label='X', y_label='Y', z_label='Z'):
    """
    Plots 3d training data

    Parameters
    ----------
    input_features : numpy.ndarray

        
    target_feature : pandas.core.series.Series
        Overrides the data type of the result.

    Returns
    ----------
    nothing - adds a matplotlib figure
    """

    fig = plt.figure(f"Training Data:{dataset_id}")
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(input_features[:,0], input_features[:,1],target_feature, alpha=0.5)
    plt.title(f'Training Data\n {dataset_id} ')
    ax.set_xlabel(x_label)
    ax.set_ylabel( y_label)
    ax.set_zlabel(z_label)

def make_3d_dataframe(data_loc:str, col_x1:int, col_x2:int, col_y:int):
    """
    makes a dataframe out of a training data csv and splits it into input features (X1, X2) and output features (y)

    Parameters
    ----------
    data_loc : str
        location of csv containing data

    col_x1 : int
        column index of X1 feature data
    
    col_x2 : int
        column index of X2 feature data

    col_y : int
        column index of y feature data

    Returns
    ----------
    X : numpy.ndarray


    Y : pandas.core.series.Series
    """
    df = pd.read_csv(data_loc)
    X1 = df.iloc[:, col_x1] 
    X2 = df.iloc[:, col_x2] 
    X = np.column_stack((X1, X2))
    Y = df.iloc[:, col_y] 

    return X, Y 

def make_poly_data(X_features, degree):
    """
    Makes polynomial products of the training data

    Parameters
    ----------
    X_features : numpy.ndarray
        features

    degree : int
        max polynomial order

    Returns
    ----------
    poly_X : numpy.ndarray

    """

    poly = PolynomialFeatures(degree)
    poly_X = poly.fit_transform(X_features)
    # print(X)
    # print(poly_X)
    # print(PolynomialFeatures(5).fit(X).get_feature_names(['X1', 'X2']))
    return poly_X

def genarate_test_values(x_upper, x_lower, y_upper, y_lower, poly_level):
    """
    Makes polynomial products of the training data

    Parameters
    ----------
    x_upper : int
        upper limit in the x dimension

    x_lower : int
        lower limit in the x dimension

    y_upper : int
        upper limit in the y dimension

    y_lower : int
        lower limit in the y dimension

    Returns
    ----------
    Xtest : numpy.ndarray 
        sample test values

    """
    Xtest = []
    x_grid = np.linspace(x_lower,x_upper, num = 20)
    y_grid = np.linspace(y_lower,y_upper, num = 20)
    for i in x_grid:
        for j in y_grid:
            Xtest.append([i,j])
    Xtest = np.array(Xtest)
    Xtest = make_poly_data(Xtest, poly_level)
    return Xtest


def plot_predictions(training_data, target_feature, test_data, predictions, t):
    """
    Plots Predictions in 3D figure

    Parameters
    ----------
    training_data : array
        array of input data

    target_feature : array
        array of target data

    test_data : array
       

    predictions : array
        

    t : array
        title of figure

    Returns
    ----------
    Nothing
        generates plot

    """
    fig = plt.figure(t)
    ax = fig.add_subplot(111, projection = '3d')
    trn = ax.scatter(training_data[:,1], training_data[:,2], target_feature, label = 'Training Data')
    # ax.plot_trisurf(training_data[:,0], training_data[:,1],target_feature)
    ax.scatter(test_data[:,1], test_data[:,2], predictions, color = 'orange')
    ### https://stackoverflow.com/questions/27449109/adding-legend-to-a-surface-plot
    fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
    # ax.scatter( test_data[:,2], test_data[:,1], predictions, color = 'orange')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend([trn, fake2Dline], ['Training Data', 'Predictions'])
    plt.title(t)

def plot_for_C(X, y, test_data, predictions, C_range):
    """
    Plots Predictions for a range of c values in 3D figures

    Parameters
    ----------
    X : array
        array of input data

    y : array
        array of target data

    test_data : array
       

    predictions : array
        

    C : array

    Returns
    ----------
    Nothing
        generates plot

    """
    for i in range(len(C_range)):
        plot_predictions(X, y, test_data, predictions[i],'C = ' + str(C_range[i]))

