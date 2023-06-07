import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt, colormaps
import matplotlib.cm as cm

from load_auto import *

df = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna()

x_train_data_all = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']]
x_train_data_horsepower = df[['horsepower']]
y_train_data = df["mpg"]

x_train_norm_horsepower = x_train_data_horsepower.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)
x_train_norm_all = x_train_data_all.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)


def initialize(dim):
    """
    :param dim: The dimensions of our weights matrix
    :return: The bias and weights that were randomly generated
    """

    bias = 0
    weights = np.zeros(dim)
    return bias, weights


def model_forward(bias, weights, x_train_data):
    """
    :param bias: From the initialize function
    :param weights: From the initialize function
    :param x_train_data: From the dataset
    :return: Calculate the predicted values
    """
    y_hat = np.dot(x_train_data, weights) + bias

    return y_hat


def compute_cost(y_train_data, y_hat):
    """
    :param y_train_data: The training values
    :param y_hat: The predicted values
    :return: cost value
    """

    y_diff = y_train_data - y_hat
    j = np.square(y_diff).mean()

    return j


def model_backward(x_train_data, y_train_data, y_hat):
    """
    :param x_train_data: From the loaded dataset
    :param y_train_data: From the loaded dataset
    :param y_hat: The calculated prediction
    :return: bias and weights derivatives
    """

    der_bias = -(np.sum(y_train_data - y_hat)) * (2 / len(y_train_data))
    der_weights = -(np.dot(x_train_data.T, (y_train_data - y_hat))) * (2 / len(y_train_data))

    return der_bias, der_weights


def update_parameters(der_bias, der_weights, bias, weights, learning_rate):
    """
    :param der_bias: bias derivative
    :param der_weights: weights derivative
    :param bias: current bias
    :param weights: current weights
    :param learning_rate: learning rate to adjust the update step
    :return: new bias and weights
    """

    new_bias = bias - der_bias * learning_rate
    new_weights = weights - der_weights * learning_rate

    return new_bias, new_weights


def predict(bias, weights, x_train_data):
    """
    Just a wrapper for model_forward function
    """
    y_hat = model_forward(bias, weights, x_train_data)

    return y_hat


def train_linear_model(x_train_data, y_train_data, learning_rate, num_iterations):
    bias, weights = initialize(x_train_data.shape[1])
    iter_num = 0
    gd_iterations_df = pd.DataFrame(columns=['iteration', 'cost'])
    result_idx = 0

    for each_iter in range(num_iterations):
        y_hat = model_forward(bias, weights, x_train_data)
        cost = compute_cost(y_train_data, y_hat)
        previous_bias = bias
        previous_weights = weights
        der_bias, der_weights = model_backward(x_train_data, y_train_data, y_hat)
        bias, weights = update_parameters(der_bias, der_weights, previous_bias, previous_weights, learning_rate)

        if iter_num % 10 == 0:
            gd_iterations_df.loc[result_idx] = [iter_num, cost]
            result_idx = result_idx + 1
        iter_num += 1
    print("Final estimate of bias and weights: ", bias, weights)

    return gd_iterations_df, bias, weights


""" PLOT THE RELATIONSHIP BETWEEN LEARNING RATE AND NUMBER OF ITERATIONS WITH ALL FEATURES AS INPUT"""
# gb_iteration_df_all_features, bias_value_all, weights_all = train_linear_model(
#     x_train_norm_all,
#     y_train_data,
#     learning_rate=1,
#     num_iterations=1000)
#
# gb_iteration_df_all_features_2, bias_value_all_2, weights_all_2 = train_linear_model(
#     x_train_norm_all,
#     y_train_data,
#     learning_rate=0.1,
#     num_iterations=1000)
#
# gb_iteration_df_all_features_3, bias_value_all_3, weights_all_3 = train_linear_model(
#     x_train_norm_all,
#     y_train_data,
#     learning_rate=0.01,
#     num_iterations=1000)
#
# gb_iteration_df_all_features_4, bias_value_all_4, weights_all_4 = train_linear_model(
#     x_train_norm_all,
#     y_train_data,
#     learning_rate=0.001,
#     num_iterations=1000)
#
# gb_iteration_df_all_features_5, bias_value_all_5, weights_all_5 = train_linear_model(
#     x_train_norm_all,
#     y_train_data,
#     learning_rate=0.0001,
#     num_iterations=1000)
#
# # plt.plot(gb_iteration_df_all_features["iteration"], gb_iteration_df_all_features["cost"], label="a = 1")
# plt.plot(gb_iteration_df_all_features_2["iteration"], gb_iteration_df_all_features_2["cost"], label="a = 0.1")
# plt.plot(gb_iteration_df_all_features_3["iteration"], gb_iteration_df_all_features_3["cost"], label="a = 0.01")
# plt.plot(gb_iteration_df_all_features_4["iteration"], gb_iteration_df_all_features_4["cost"], label="a = 0.001")
# plt.plot(gb_iteration_df_all_features_5["iteration"], gb_iteration_df_all_features_5["cost"], label="a = 0.0001")
# plt.legend()
# plt.title("Relationship between cost and learning rate with all features as an input")
# plt.ylabel("Cost")
# plt.xlabel("Number of iterations")
# plt.show()

""" PLOT THE RELATIONSHIP BETWEEN LEARNING RATE AND NUMBER OF ITERATIONS WITH HORSEPOWER AS INPUT"""
# gb_iteration_df_horsepower, bias_value_horsepower, weights_horsepower = train_linear_model(
#     x_train_norm_horsepower,
#     y_train_data,
#     learning_rate=1,
#     num_iterations=1000)
#
# gb_iteration_df_horsepower_2, bias_value_horsepower_2, weights_horsepower_2 = train_linear_model(
#     x_train_norm_horsepower, y_train_data,
#     learning_rate=0.1,
#     num_iterations=1000)
#
# gb_iteration_df_horsepower_3, bias_value_horsepower_3, weights_horsepower_3 = train_linear_model(
#     x_train_norm_horsepower, y_train_data,
#     learning_rate=0.01,
#     num_iterations=1000)
#
# gb_iteration_df_horsepower_4, bias_value_horsepower_4, weights_horsepower_4 = train_linear_model(
#     x_train_norm_horsepower, y_train_data,
#     learning_rate=0.001,
#     num_iterations=1000)
#
# gb_iteration_df_horsepower_5, bias_value_horsepower_5, weights_horsepower_5 = train_linear_model(
#     x_train_norm_horsepower, y_train_data,
#     learning_rate=0.0001,
#     num_iterations=1000)
#
# plt.plot(gb_iteration_df_horsepower["iteration"], gb_iteration_df_horsepower["cost"], label="a = 1")
# plt.plot(gb_iteration_df_horsepower_2["iteration"], gb_iteration_df_horsepower_2["cost"], label="a = 0.1")
# plt.plot(gb_iteration_df_horsepower_3["iteration"], gb_iteration_df_horsepower_3["cost"], label="a = 0.01")
# plt.plot(gb_iteration_df_horsepower_4["iteration"], gb_iteration_df_horsepower_4["cost"], label="a = 0.001")
# plt.plot(gb_iteration_df_horsepower_5["iteration"], gb_iteration_df_horsepower_5["cost"], label="a = 0.0001")
# plt.legend()
# plt.title("Relationship between cost and learning rate with horsepower as an input")
# plt.ylabel("Cost")
# plt.xlabel("Number of iterations")
# plt.show()

""" EVALUATE THE PERFORMANCE OF THE MODEL"""
gb_iteration_df_horsepower, bias_value_horsepower, weights_horsepower = train_linear_model(x_train_norm_horsepower,
                                                                                           y_train_data,
                                                                                           learning_rate=0.01,
                                                                                           num_iterations=1001)

print(gb_iteration_df_horsepower.iloc[-1:])

gb_iteration_df_all_features, bias_value_all_features, weights_all_features = train_linear_model(x_train_norm_all,
                                                                                                 y_train_data,
                                                                                                 learning_rate=0.01,
                                                                                                 num_iterations=1001)

print(gb_iteration_df_all_features.iloc[-1:])

""" MODEL PREDICTIONS WITH HORSEPOWER AS AN INPUT"""
plt.scatter(x_train_norm_horsepower, y_train_data, c=y_train_data, cmap="Spectral")
plt.plot(x_train_norm_horsepower, (weights_horsepower * x_train_norm_horsepower) + bias_value_horsepower, color="black")
plt.ylabel("mpg")
plt.xlabel("Horsepower")
plt.show()
