from parameters import Parameters


class DiffusionCoefficient:
    def __init__(self, params: Parameters):
        self.params = params

    def get_D_i_j(self, T, M_i, M_j, y_i, y_j):
        return 1e-3 * (T ** 1.75) * (((M_i ** -1) + (M_j ** -1)) ** 0.5) / (self.params.p_t * 0.98692327 * ())

    def get_D_i_m(self, y_i, y_j, D_i_j):
        return (1 - y_i) / ()

    def get_D_i_Kn(self):

    def get(self, r, T):
        return self.params.epsilon / self.params.tau * (1 / self.get_D_i_m() + self.get_D_Kn)
