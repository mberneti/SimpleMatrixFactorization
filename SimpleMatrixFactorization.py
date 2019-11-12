from plotAnimator import plotAnimator
import numpy as np
"""
A Simple Implementation in Python
Once we have derived the update rules as described above, 
it actually becomes very straightforward to implement the algorithm. 
The following is a function that implements the algorithm in Python 
using the stochastic gradient descent algorithm. Note that this implementation requires the Numpy module.
"""


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

        # real user as x, item as y and rate as r and rpredict
        xs, ys = self.R.nonzero()
        self.z = [self.R[x, y] for x, y in zip(xs, ys)]
        self.x = [0 for i in range(len(self.z))]
        self.y = [i for i in range(len(self.z))]
        self.realLine = np.array([self.x, self.y, self.z])
        self.log = [self.realLine]
        self.shiftXIndex = 0

    def getShiftX(self):
        self.shiftXIndex += .5
        return [self.shiftXIndex for i in range(len(self.z))]

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(
            scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(
            scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        # general_mean of non zero values
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        # ----------------------------
        newPredictLine = np.array(
            [self.getShiftX(), self.y, [predicted[x, y] for x, y in zip(xs, ys)]])

        self.log.append(newPredictLine)
        # ----------------------------
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            # self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            # self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            self.b_u[i] += self.alpha * (2*e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (2*e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * \
                (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * \
                (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + \
            self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


R = np.array([
    [5, 3, 0, 1, 3],
    [4, 0, 0, 1, 3],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 2],
    [0, 1, 5, 4, 1],
])

# R = np.random.randint(5, size=(10, 10))

mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=30)

mf.train()

realLine = mf.log[0]

p = plotAnimator(mf.log[1:])

# draw real line -----------------------------------------
p.ax.plot3D(realLine[0], realLine[1], realLine[2], 'grey')


# x = np.array([[0, 10] for i in range(6)])
# y = np.array([[0, 10] for i in range(6)])

# z = realLine[2][1:].reshape(6, 2)
# print(x, y, z)
# p.ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
# p.ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
# camera
p.ax.view_init(elev=5., azim=0)
p.plt.show()
