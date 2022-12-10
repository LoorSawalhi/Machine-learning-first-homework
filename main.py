from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import random


def clean_data(data):
    for x in data.columns:
        summation = 0
        count = 0
        for y in data[x]:
            if y > 0:
                summation += y
                count += 1

        mean = summation / count

        for y in data[x]:
            if y == 0:
                y = mean

        if x != "Final":
            data[x] = (data[x] - data[x].mean()) / data[x].std()

    return data


def linear_regression(train_data, train_output):
    data_array = np.array(train_data)
    data_transpose = np.transpose(data_array)
    output_array = np.array(train_output)

    weights_vector = np.linalg.inv(np.dot(data_transpose, data_array))
    weights_vector = np.dot(weights_vector, data_transpose)
    weights_vector = np.dot(weights_vector, output_array)

    return weights_vector


def linear_regression_lib(train_data, train_output, data_test):
    model_lin = LinearRegression()
    model_lin.fit(train_data, train_output)
    predict_output = model_lin.predict(data_test)

    return predict_output


def gradient_descent(train_data, train_output):
    alpa = 0.0001
    training_data_array = np.array(train_data)
    learned_weights = np.zeros(len(np.transpose(training_data_array)))
    n = len(np.transpose(training_data_array))

    for x in range(0, 10000):
        y_predicted = np.dot(training_data_array, learned_weights)
        error = y_predicted - train_output
        gradient = np.dot(np.transpose(training_data_array), error)
        learned_weights = learned_weights - alpa * gradient

    return learned_weights


def sum_square_error(label, output_test):
    sse = (label - output_test) ** 2
    sse = np.sum(sse)
    return sse


data = pd.read_csv("grades.csv", sep=",")
data = clean_data(data)

for x in data.columns:
    if abs(data[x].corr(data["Final"])) < 0.8:
        data = data[data.columns.drop(x)]

data["Constant"] = 1
Training = data.iloc[random.sample(range(0, len(data)), int(len(data) * 0.8))]
Testing = data.drop(Training.index)
Training = Training.reset_index(drop=True)
Testing = Testing.reset_index(drop=True)

training_data = Training[data.columns.drop('Final')]
training_output = Training['Final']
test_data = Testing[data.columns.drop('Final')]
test_output = Testing['Final']

weights = linear_regression(training_data, training_output)
labels = np.dot(test_data, weights)

print(sum_square_error(labels, test_output))

predicted_output = linear_regression_lib(training_data, training_output, test_data)

print(sum_square_error(predicted_output, test_output))

learning_weights = gradient_descent(training_data, training_output)
labels = np.dot(test_data, learning_weights)

print(sum_square_error(labels, test_output))
