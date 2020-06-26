from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.data import Dataset
import logging


logger = tf.get_logger()
logger.setLevel(logging.ERROR)


pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


housing_df = pd.read_csv("train_LR.csv", sep=",")
housing_df = shuffle(housing_df)
housing_df.head()

# Processing the data
processed_features = housing_df[["GrLivArea"]]
output_targets = housing_df[["SalePrice"]]


# Splitting the data into training, validation, and test sets
training_examples = processed_features[0:1060]
training_targets = output_targets[0:1060]

"""
The model is fit on the training set, 
and the fitted model is used to predict
the responses for the observations in the validation set.
"""

# 200 values for validation
val_examples = processed_features[1060:1260]
val_targets = output_targets[1060:1260]

# 200 values for testing
test_examples = processed_features[1260:1460]
test_targets = output_targets[1260:1460]

# Configure a numeric feature column for GrLivArea.
my_feature_columns = [tf.feature_column.numeric_column("GrLivArea")]

# Define the preferred optimizer: in this case lets use gradient descent
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
model = tf.estimator.LinearRegressor(
    feature_columns=my_feature_columns, optimizer=my_optimizer
)

print(my_feature_columns)

# Define the input function required for training
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


# Train the model from the existing data
training = model.train(
    input_fn=lambda: my_input_fn(training_examples, training_targets), steps=1000
)
