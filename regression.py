import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    x = (response.text.split('\n'))
    x[0] = x[0].split(',')
    x[1] = x[1].split(',')
    n = len(x[0])
    X = numpy.array(([float(i) for i in x[0] if i!='area'], numpy.ones(n-1)))
    Y = numpy.array([float(i) for i in x[1] if i!='price'])
    z = numpy.linalg.pinv(X.dot(X.T))
    y = X.dot(Y)
    theta = z.dot(y)
    theta.reshape(2,1)
    x = numpy.array((area, numpy.ones(len(area))))
#     print(x.shape, )
    return x.T.dot(theta)     
    
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
