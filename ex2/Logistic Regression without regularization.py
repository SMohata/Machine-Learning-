from numpy import *
from matplotlib.pyplot import *

""" We will implement logistic regression and apply it to two diﬀerent datasets.
In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university. Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.

Your task is to build a classiﬁcation model that estimates an applicant’s probability of admission based the scores from those two exams."""


# loading data
data = loadtxt('ex2data1.txt', delimiter=",")
x = data[:, :2]
y = c_[data[:, 2]]  # 100x1

# no. of training sets
m = len(y)  # 100

# adding a column of biases to the x matrix and initialzing other variables
X = c_[ones(m), x]  # 100x3

# no. of features
n = len(X[1])  # 3

"""Before starting to implement any learning algorithm, it is always good to visualize the data if possible. The figure depicts the axes as the two exam scores, and the positive and negative examples are shown with diﬀerent markers."""


def Plot(x, y):
    pos = where(y == 1)
    neg = where(y == 0)
    scatter(x[pos, 0], x[pos, 1], marker='+', c='purple')
    scatter(x[neg, 0], x[neg, 1], marker='o', c='y')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    title('Scatter Plot of Training data')
    legend(['Admitted', 'Not admitted'])
    show()


# Initialize fitting parameters
theta = zeros((n, len(y[1])))  # 3x1

# Initialize number of iterations to achieve gradient descent
iter = 1500

# Initializing learning rate for Gradient Descent
alpha = 0.01

"""Before you start with the actual cost function, recall that the logistic regression hypothesis is deﬁned as sigmoid function.Your ﬁrst step is to implement this function so that it can be called by the rest of your program. """


def sigmoid(z):
    h = 1 / (1 + exp(-z))
    return h


"""Now we will implement the cost function and gradient for logistic regression.Note that while this gradient looks identical to the linear regression gradient, the formula is actually diﬀerent because linear and logistic regression have diﬀerent deﬁnitions of hθ(x)."""
"""The function calculates the cost function to be minimized which is the square of the difference between the actual output and the predicted output divided by the number of training exams times two. First, the weights are multiplied by X to find the predicted output. y is then subtracted from this predicted output. and the mean error and cost function is computed"""


def get_cost(X, y, theta):
    h = sigmoid(X.dot(theta))
    cost = sum(-((y * log(h)) + (1 - y) * log(1 - h))) / m
    return cost


"""The function computes the gradient descent algorithm that is constantly updating the value of theta so that the cost function is minimized."""


def gradient_descent(X, y, theta):
    for i in range(iter):
        hyp = X.dot(theta)
        error = sigmoid(hyp) - y
        theta = theta - ((alpha / m) * (X.T.dot(error)))
    return theta


print("Visualize Data...")
Plot(x, y)

print("The initial cost is: %f" % get_cost(X, y, theta))
print("\nTheta obtained after gradient descent")
new_theta = gradient_descent(X, y, theta)
print(new_theta)


"""After learning the parameters, you can use the model to predict whether a particular student will be admitted. c"""


def predict(new_theta, input1, input2):
    """The function predicts the output using weights obtained from gradient descent"""
    X = c_[ones(1), input1, input2]
    return (sigmoid(X.dot(new_theta)))


print("For a student with an Exam 1 score of 45 and an Exam 2 score of 85, admission probability is: ", float(predict(new_theta, 45, 85)))
