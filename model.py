import numpy as np

from consts import *

def dgdP(P):
    return np.block([
        [np.identity(2) * 1/mu_s, np.zeros((2, 2))],
        [np.zeros((2, 2)), np.identity(2) * 1/mu_f]
    ])

def dfdQ(Q):
    def A(r):
        mag = np.linalg.norm(r)
        return (mag**2 * np.identity(2) - 3 * (r * r.T)) / mag**5

    # Note that Q[0] = Q_1, Q[1] = Q_3 as Q_2 isn't stored (COM of solar system)
    A_1 = M_s * M_e * A(Q[0] + s * Q[1])
    A_2 = M_s * M_m * A(Q[0] + (s - 1) * Q[1])

    return np.block([
        [-A_1 - A_2, -s * A_1 + (1 - s) * A_2],
        [-s * A_1 + (1 - s) * A_2, -s**2 * A_1 - (s - 1)**2 * A_2 - M_e * M_m * A(Q[1])]
    ])

def a(r):
    return r / np.linalg.norm(r)**3

def f(Q):
    a1 = M_s * M_e * a(Q[0] + s * Q[1])
    a2 = M_s * M_m * a(Q[0] + (s - 1) * Q[1])
    return np.block([
        [-a1 - a2],
        [-s * a1 + (1 - s) * a2 - M_e * M_m * a(Q[1])]
    ])

def g(P):
    return np.block([
        [P[0] / mu_s],
        [P[1] / mu_f]
    ])


class State:
    def __init__(self, Q_0, P_0):
        self.Q = Q_0
        self.P = P_0

    def __repr__(self):
        return "State(Q: {%s}, P: {%s})" % (self.Q, self.P)

    def to_vector(self):
        return np.concatenate((self.Q.flatten(), self.P.flatten())).T
    def get_Q(self):
        return self.Q
    def get_P(self):
        return self.P


class Model:
    def __init__(self, dt):
        self.dt = dt

        # Start conditions
        self.DQ_n = np.block([
            [np.identity(4), np.zeros((4, 4))]
        ])
        self.DP_n = np.block([
            [np.zeros((4, 4)), np.identity(4)]
        ])

    # -- Private methods --
    def __DP_n_p_half(self, x):
        return self.DP_n + self.dt/2 * np.block([dfdQ(x.get_Q()), np.zeros((4, 4))]) * self.DQ_n

    def __DQ_n_p_1(self, P_n_p_half, DP_n_p_half):
        return self.DQ_n + self.dt * np.block([np.zeros((4, 4)), dgdP(P_n_p_half)]) * DP_n_p_half

    def __DP_n_p_1(self, DP_n_p_half, Q_n_p_1, DQ_n_p_1):
        return DP_n_p_half + self.dt/2 * np.block([dfdQ(Q_n_p_1), np.zeros((4, 4))]) * DQ_n_p_1

    # TODO: Implement f & g
    def __P_n_p_half(self, x):
        return x.get_P() + self.dt/2 * f(x.get_Q())

    def __Q_n_p_1(self, x, P_n_p_half):
        return x.get_Q() + self.dt * g(P_n_p_half)

    """
    def __P_n_p_1(self, P_n_p_half, Q_n_p_1):
        return P_n_p_half + self.dt/2 * __f(Q_n_p_1)
    """

    def __calc_M(self, x):
        print("Calculating M for state: ", x)

        P_n_p_half = self.__P_n_p_half(x)
        Q_n_p_1 = self.__Q_n_p_1(x, P_n_p_half)

        DP_n_p_half = self.__DP_n_p_half(x)
        DQ_n_p_1 = self.__DQ_n_p_1(P_n_p_half, DP_n_p_half)

        return np.block([
            [DQ_n_p_1],
            [self.__DP_n_p_1(DP_n_p_half, Q_n_p_1, DQ_n_p_1)]
        ])

    # -- Public --
    def step(self, x):
        """
        Step the model.
        Here "x" is (Q_n^T, P_n^T)^T -> vector of system state
        """
        M = self.__calc_M(x)
        print("Calculated M: ", M)
        print("M SHAPE: ", M.shape)
        x_vec = x.to_vector()
        print("X SHAPE: ", x_vec.shape)

        return np.matmul(M, x_vec)

    def step_AD(self, x, M=None):
        """
        Adjoint step. M is optional in case it's already computed from linear run.
        """
        if M == None:
            M = self.__calc_M(x)
        x_vec = x.to_vector()

        return np.matmul(M.T, x_vec)

    def finalise(self):  # Save to file?
        pass



if __name__ == "__main__":
    m = Model(0.1)
    Q_start = np.array([[0.1, 0.1], [1.0, 1.0]])
    P_start = np.zeros((2, 2))

    x0 = State(Q_start, P_start)

    x1 = m.step(x0)
    print("x: ", x0)
    print("x stepped: ", x1)

