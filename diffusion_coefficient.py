import casadi as ca


class DiffusionCoefficient:

    @staticmethod
    def get_D_i_j(T, p, M_i, M_j, v_i, v_j):
        return 1e-1 * (T ** 1.75) * (((M_i ** -1) + (M_j ** -1)) ** 0.5) / (
                p * 0.98692327 * ((v_i ** (1 / 3)) + (v_j ** (1 / 3))) ** 2)

    @staticmethod
    def get_D_i_m(y_i, y_j_1, y_j_2, y_j_3, D_i_j_1, D_i_j_2, D_i_j_3):
        return (1 - y_i) / ((y_j_1 / D_i_j_1) + (y_j_2 / D_i_j_2) + (y_j_3 / D_i_j_3))

    @staticmethod
    def get_D_i_Kn(ctx, M_i):
        return 1e6 * ctx.params.d_pore / 3 * ca.sqrt(8e3 * ctx.params.R * ctx.T / (ca.pi * M_i))