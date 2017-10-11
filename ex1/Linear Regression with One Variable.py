from numpy import *
from matplotlib.pyplot import *

"""We will implement linear regression with one variable to predict proﬁts for a food truck. Suppose you are the CEO of a restaurant franchise and are considering diﬀerent cities for opening a new outlet. The chain already has trucks in various cities and you have data for proﬁts and populations from the cities.

The ﬁle ex1data1.txt contains the dataset for our linear regression problem. The ﬁrst column is the population of a city and the second column is the proﬁt of a food truck in that city. A negative value for proﬁt indicates a loss."""

# loading comma separated data
data = loadtxt('ex1data1.txt', delimiter=",")
x = data[:, 0]
y = c_[data[:, 1]]

# no. of training sets
m = len(y)  # 97

# Adding a column of biases to the x matrix
X = c_[ones(m), x]

# no. of features in the data
n = X.shape[1]  # 2

"""Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (proﬁt and population)."""


def Plot(x, y):
    plot(x, y, 'rx')
    ylabel('Profit in $10,000s')
    xlabel('Population of City in 10,000s')
    title('Profit')
    show()


# Initialize fitting parameters
theta = zeros((n, len(y[1])))  # 2x1

# Initialize number of iterations to achieve gradient descent
iter = 1500

# Initializing learning rate for Gradient Descent
alpha = 0.01

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
        hyp = X.dot(theta)
        error = hyp - y
        theta = theta - ((alpha / m) * (X.T.dot(error)))
        J = get_cost(X, y, theta)
        temp.append(J)
    return temp, theta


print("Visualize Data...")
Plot(x, y)

print("The initial cost is: %f" % get_cost(X, y, theta))
print("\nTheta obtained after gradient descent")
new_theta = gradient_descent(X, y, theta)[1]
print(new_theta)
print("Plotting line of best fit")
plot(x, y, 'rx')
ylabel('Profit in $10,000s')
xlabel('Population of City in 10,000s')
prediction = X.dot(new_theta)
plot(x, prediction, '-')
title('Line of Best Fit')
show()

"""To visualize the gradient descent algorithm we can plot the cost function computed after every iteration and we observe a graph where as the number of iterations increase, the cost function saturates"""

cost = gradient_descent(X, y, theta)[0]
plot(cost, "-")
ylabel("Cost Function")
xlabel("No. of iterations")
title('Working of Gradient Descent')
show()


# Predict values for population sizes of 35,000 and 70,000

def predict(new_theta, input):
    """The function predicts the output using weights obtained from gradient descent"""
    X = c_[ones(1), input]
    return (X.dot(new_theta))


print("For population = 35,000, we predict a profit of %f" % (predict(new_theta, 3.5) * 10000))
print("For population = 70,000, we predict a profit of %f" % (predict(new_theta, 7) * 10000))
