# Training a basic neural network using linear regression and sklearn
# Author: Omar Barazanji
# Date: May 2020
# Sources: Aurelien Geron's book on ML.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os

HOUSING_PATH = "datasets/housing"

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    housing = load_housing_data()
    # calling housing.info() in python -i mode will reveal the csv details
    #
    # housing["ocean_proximity"].value_counts() makes output below:
    # <1H OCEAN     9136
    # INLAND        6551
    # NEAR OCEAN    2658
    # NEAR BAY      2290
    # ISLAND           5
    # Name: ocean_proximity, dtype: int64
    #
    # calling housing.describe will show mean/std/ other stats on data.
    # Let's plot a histogram of some of the data to better see what we have:

    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # We want to split 20% of the data aside for the beginning:
    # >>> train_set, test_set = split_train_test(housing, 0.2)
    # >>> print(len(train_set), "train +", len(test_set), "test")
    # 16512 train + 4128 test

    # The above works for one instance of the test data, but it is
    # not going to be consistent for each time you want it.
    # So we do this instead!

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # >>> housing["income_cat"].value_counts() / len(housing)
    # 3.0    0.350581
    # 2.0    0.318847
    # 4.0    0.176308
    # 5.0    0.114438
    # 1.0    0.039826

    # remove income_cat attribute to put data back to original state
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    housing = strat_train_set.copy()  # create a copy to play with!
    # housing.plot(kind="scatter", x="longitude", y="latitude")
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"),
                 colorbar=True,) # tells us housing prices related to location.
    # plt.legend()
    # plt.show()  # really cool !!

    # Let's get to correlating!

    corr_matrix = housing.corr()

    # Let's correlate our housing data with median house value!
    # >>> corr_matrix["median_house_value"].sort_values(ascending=False)
    # median_house_value    1.000000 (1.0 is good sign we fed correct data in!)
    # median_income         0.687160 (good corr coeff!)
    # total_rooms           0.135097
    # housing_median_age    0.114110
    # households            0.064506
    # total_bedrooms        0.047689
    # population           -0.026920
    # longitude            -0.047432
    # latitude             -0.142724

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))

    # plt.show() # median income -> good predictor for median house value.
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
    # plt.show() PDF page: 108 (where i left off )

    housing = strat_train_set.drop("median_house_value", axis=1) # drop creates
    # a copy without affecting strat_train_set

    housing_labels = strat_train_set["median_house_value"].copy()

    # total_bedrooms attribute has some missing values, so let’s fix this.
    # We have three options:
    # Get rid of the corresponding districts.
    # Get rid of the whole attribute.
    # Set the values to some value (zero, the mean, the median, etc.).

    # Scikit learn has good class for handling missing values: The Imputer.
    imputer = SimpleImputer(strategy="median")

    # Since the median can only be computed on numerical attributes, we need to
    # create a copy of the data without the text attribute ocean_proximity:
    # - Aurelien Geron (pdf pg. 110)
    housing_num = housing.drop("ocean_proximity", axis=1)

    # Now you can fit the imputer instance to the training data using the fit()
    # method:
    imputer.fit(housing_num)

    # >>> imputer.statistics_
    # array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
    #         408.    ,    3.5409])

    # >>> housing_num.median().values
    # array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
    #         408.    ,    3.5409])

    # Now you can use this “trained” imputer to transform the training set by
    # replacing missing values by the learned medians
    X = imputer.transform(housing_num)
    # The result is a plain Numpy array containing the transformed features. If you
    # want to put it back into a Pandas DataFrame, it’s simple:

    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

    # Handling Text (ocean proximity)
    # Scikit-Learn provides a transformer for this task called LabelEncoder
    encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)

    # >>> housing_cat_encoded
    # array([0, 0, 4, ..., 1, 0, 3])
    # >>> print(encoder.classes_)
    # ['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN']

    # Let’s encode the categories as one-hot vectors. Note
    # that fit_transform() expects a 2D array, but housing_cat_encoded is a 1D
    # array, so we need to reshape it:
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
    # output is a sparse matrix (saves data by not using zeros)
    # to convert use to_array() to make normal numpy array...

    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)

    # >>> housing_cat_1hot
    # array([[1, 0, 0, 0, 0],
    #    [1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1],
    #    ...,
    #    [0, 1, 0, 0, 0],
    #    [1, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 0]])

# From book:
#     Standardization is quite different: first it subtracts the mean value (so
# standardized values always have a zero mean), and then it divides by the
# variance so that the resulting distribution has unit variance. Unlike min-max
# scaling, standardization does not bound values to a specific range, which may be
# a problem for some algorithms (e.g., neural networks often expect an input value
# ranging from 0 to 1). However, standardization is much less affected by outliers.
# For example, suppose a district had a median income equal to 100 (by mistake).
# Min-max scaling would then crush all the other values from 0–15 down to 0–
# 0.15, whereas standardization would not be much affected. Scikit-Learn provides
# a transformer called StandardScaler for standardization.

# left off at page 117
