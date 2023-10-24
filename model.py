import numpy as np

from consts import *

def __dgdP(P):
    return np.block([
        [np.identity(2) * 1/mu_s, np.zeros((2, 2))],
        [np.zeros((2, 2)), np.identity(2) * 1/mu_f]
    ])

def __dfdQ(Q):
    def A(r):
        mag = r.norm()
        return (mag**2 * np.identity(2) - 3 * (r * r.T)) / mag**5

    # Note that Q[0] = Q_1, Q[1] = Q_3 as Q_2 isn't stored (COM of solar system)
    A_1 = M_s * M_e * A(Q[0] + s * Q[1])
    A_2 = M_s * M_m * A(Q[0] + (s - 1) * Q[1])

    return np.block([
        [-A_1 - A_2, -s * A_1 + (1 - s) * A_2],
        [-s * A_1 + (1 - s) * A_2, -s**2 * A_1 - (s - 1)**2 * A_2 - M_e * M_m * A(Q[1])]
    ])


def State:
    def __init__(self, Q_0, P_0):
        self.x = np.block([
            [Q_0.T],
            [P_0.T]
        ])

    def get_Q(self):
        return self.x[:self.x.len()//2]
    def get_P(self):
        return self.x[self.x.len()//2:]

def Q_n_from_state(x):

def P_n_from_state(x):

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
        return self.DP_n + self.dt/2 * __dfdQ(x.get_Q()) * self.DQ_n

    def __DQ_n_p_1(self, x, DP_n_p_half):
        return self.DQ_n + self.dt * __dgdP(x.get_P()) * DP_n_p_half

    # TODO: Implement f & g
    def __P_n_p_half(self, x):
        return x.get_P() + self.dt/2 * __f(x.get_Q())

    def __Q_n_p_1(self, x, P_n_p_half):
        return x.get_Q() + self.dt * __g(P_n_p_half)

    def __P_n_p_1(self, x, P_n_p_half, Q_n_p_1):
        return P_n_p_half + self.dt/2 * __f(Q_n_p_1)

    def __calc_M(self):
        return np.block([
            []
        ])

    # -- Public --
    def step(self, x):
        """
        Step the model.
        Here "x" is (Q_n^T, P_n^T)^T -> vector of system state
        """
        M = self.__calc_M()
        return M * x

    def step_AD(self, x):
        pass

    def finalise(self):  # Save to file?
        pass



if __name__ == "__main__":
    m = Model()
    print(m.DQ_n)

