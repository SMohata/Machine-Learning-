from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import *
from numpy.linalg import inv

"""We will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to ﬁrst collect information on recent houses sold and make a model of housing prices.

The ﬁle ex1data2.txt contains a training set of housing prices in Portland, Oregon. The ﬁrst column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house. """

# loading comma separated data
data = loadtxt('ex1data2.txt', delimiter=",")
x = data[:, :2]  # load dataset excluding 2 column
y = c_[data[:, 2]]

# no. of training sets
m = len(y)  # 47

"""Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a 3-D plot to visualize the data, since it has three properties to plot (size, no. of bedrooms and price)."""


def Plot():
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')  # adding an additional axes for a three dimensional projection
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    xlabel('Size of the House')
    ylabel('Number of Bedrooms')
    ax.set_zlabel('Price of the house')
    ax.scatter(xs, ys, zs)
    show()


# Initialize number of iterations to achieve gradient descent
iter = 1500

# Initializing learning rate for Gradient Descent
alpha = 0.01

""" By looking at the values, note that house sizes are about 1000 times the number of bedrooms. When features diﬀer by orders of magnitude, ﬁrst performing feature scaling can make gradient descent converge much more quickly.

Feature Normalising the data"""
print("Before normalisation x is:")
print(x)


def feature_normalize(X):
    """Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when working with learning algorithms."""
    mean_r = []  # initialising an empty list which can be appended later
    std_r = []
    X_norm = X
    for i in range(len(X[1])):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s
    return X_norm, mean_r, std_r


print("After normalisation x is:")
print(feature_normalize(x)[0])

x, mean_r, std_r = feature_normalize(x)

# Adding a column of biases to the x matrix
X = c_[ones(m), x[:, :2]]

# no. of features in the data
n = len(X[1])  # 3

# Initialize fitting parameters
theta = zeros((n, len(y[1])))  # 3x1

"""Calculating the Gradient Descent"""

"""The function calculates the cost function to be minimized which is the square of the difference between the actual output and the predicted output divided by the number of training exams times two. First, the weights are multiplied by X to find the predicted output. y is then subtracted from this predicted output. and the mean error and cost function is computed"""


def get_cost(X, y, theta):
    hyp = X.dot(theta)  # hyp represents the hypothesis which is calculated by multiplying the X and theta matrix
    error = hyp - y  # the difference between the predicted value an the actual value is calculated
    sqr_error = error**2  # squared error
    J = sum(sqr_error) / (2 * m)  # The cost function is calculated as per the formula and the sum function is used to convert the sqr_error matrix to a numerical value
    return J


"""The function computes the gradient descent algorithm that is constantly updating the value of theta so that the cost function is minimized."""


def gradient_descent(X, y, theta):
    temp = []  # creating an empty list to store values of theta in order to update it later
    for i in range(0, iter):
        hyp = X.dot(theta)  # hyp represents the hypothesis which is calculated by multiplying the X and theta matrix
        error = hyp - y  # the difference between the predicted value an the actual value is calculated
        theta = theta - ((alpha / m) * (X.T.dot(error)))  # The cost function is calculated as per the formula
        J = get_cost(X, y, theta)
        temp.append(J)  # The cost function is updated with the new theta
    return temp, theta


print("Visualize Data...")
Plot()

print("The initial cost is: %f" % get_cost(X, y, theta))
print("\nTheta obtained after gradient descent")
new_theta = gradient_descent(X, y, theta)[1]
print(new_theta)

"""To visualize the gradient descent algorithm we can plot the cost function computed after every iteration and we observe a graph where as the number of iterations increase, the cost function saturates"""

cost = gradient_descent(X, y, theta)[0]
plot(cost)
ylabel("Cost Function")
xlabel("No. of iterations")
title('Working of Gradient Descent')
show()

"""The function predicts the value by using normal equation"""


def normal_equation(X, y):
    n = ((inv(X.T.dot(X))).dot((X.T).dot(y)))
    return n


print("Theta obtained after Normal Equation")
print(normal_equation(X, y))


"""The function predicts the value of the new output by using weights from gradient descent"""


def predict(weights, input1, input2):
    X_norm = c_[ones(1), ((input1 - mean_r[0]) / std_r[0]), ((input2 - mean_r[1]) / std_r[1])]
    print(X_norm)
    return(X_norm.dot(weights))


print("For a house of size 1650 with 3 bedrooms, we predict a price of %0.2f" % (predict(new_theta, 1650.0, 3.0)))
print("For a house of size 1650 with 3 bedrooms, we predict a price of %0.2f" % (predict(normal_equation(X, y), 1650, 3)))
