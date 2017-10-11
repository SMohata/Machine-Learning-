from numpy import *
from matplotlib.pyplot import *

"""In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly. Suppose you are the product manager of the factory and you have the test results for some microchips on two diﬀerent tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
"""

# loading data
data = loadtxt('ex2data2.txt', delimiter=",")
x = data[:, :2]
y = c_[data[:, 2]]

# no. of training sets
m = len(y)  # 118

# adding a column of biases to the x matrix and initialzing other variables
X = c_[ones(m), x]  # 118x3

# no. of features
n = len(X[1])  # 3

"""Before starting to implement any learning algorithm, it is always good to visualize the data if possible. The figure depicts the axes as the two exam scores, and the positive and negative examples are shown with diﬀerent markers."""


def Plot(x, y):
    pos = where(y == 1)
    neg = where(y == 0)
    scatter(x[pos, 0], x[pos, 1], marker='+', c='purple')
    scatter(x[neg, 0], x[neg, 1], marker='o', c='y')
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    title('Scatter Plot of Training data')
    legend(['Accepted', 'Rejected'])
    show()


# Initialize fitting parameters
theta = zeros((n, len(y[1])))  # 3x1

# Initialize number of iterations to achieve gradient descent
iter = 1500

# Initializing learning rate for Gradient Descent
alpha = 0.01

# Initializing regularization parameter for regularization
l = 1

"""Before you start with the actual cost function, recall that the logistic regression hypothesis is deﬁned as sigmoid function.Your ﬁrst step is to implement this function so that it can be called by the rest of your program. """


def sigmoid(z):
    h = 1 / (1 + exp(-z))
    return h


""" In the next parts of the exercise, you will implement regularized logistic regression to ﬁt the data and also see for yourself how regularization can help combat the overﬁtting problem.
Now you will implement code to compute the cost function and gradient for regularized logistic regression. """


def get_cost(X, y, theta):
    h = sigmoid(X.dot(theta))
    cost = sum((y * log(h)) + (1 - y) * log(1 - h)) / (-m)
    return cost


def regularised_cost(X, y, theta):
    J = get_cost(X, y, theta) + ((l / 2 * m) * sum(theta[:, 0]**2))
    return J


"""The function computes the gradient descent algorithm that is constantly updating the value of theta so that the cost function is minimized."""


def regularised_gradient_descent(X, y, theta):
    for i in range(iter):
        hyp = sigmoid(X.dot(theta))  # hypothesis is computed
        theta1 = theta[0, 0]
        grad = X.T.dot(hyp - y)
        theta[0, 0] = theta1 - ((alpha / m) * (grad[0, 0]))  # For the first value of theta regularised parameters are not calculated
        theta[1:, 0] = theta[1:, 0] - ((alpha / m) * (grad[1:, 0])) - (l / m) * theta[1:, 0]
    return theta


print(regularised_gradient_descent(X, y, theta))
print("Visualize Data...")
Plot(x, y)

print("The initial cost is: %f" % get_cost(X, y, theta))
print("The regularized cost is: %f" % regularised_cost(X, y, theta))
print("\nTheta obtained after regularised gradient descent")
new_theta = regularised_gradient_descent(X, y, theta)
print(new_theta)


"""After learning the parameters, you can use the model to predict whether a particular microchip is accepted or rejected."""


def predict(new_theta, input1, input2):
    """The function predicts the output using weights obtained from gradient descent"""
    X = c_[ones(1), input1, input2]
    return (sigmoid(X.dot(new_theta)))


print("For a microchip with test result of -0.25 and 1.5 acceptance probability is: ", float(predict(new_theta, -0.25, 1.5)))
