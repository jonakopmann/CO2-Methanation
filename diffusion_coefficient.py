import casadi as ca

from context import Context


class DiffusionCoefficient:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.params = ctx.params
        self.D_co2_h2 = 0
        self.D_co2_ch4 = 0
        self.D_co2_h2o = 0
        self.D_h2_h2o = 0
        self.D_h2_ch4 = 0
        self.D_ch4_h2o = 0

    @staticmethod
    def get_D_i_j(T, p, M_i, M_j, v_i, v_j):
        return 1e-1 * (T ** 1.75) * (((M_i ** -1) + (M_j ** -1)) ** 0.5) / (
                p * 0.98692327 * ((v_i ** (1 / 3)) + (v_j ** (1 / 3))) ** 2)

    @staticmethod
    def get_D_i_m(y_i, y_j_1, y_j_2, y_j_3, D_i_j_1, D_i_j_2, D_i_j_3):
        return (1 - y_i) / ((y_j_1 / D_i_j_1) + (y_j_2 / D_i_j_2) + (y_j_3 / D_i_j_3))

    def get_D_i_Kn(self, T, M_i):
        return self.params.d_pore / 3 * ca.sqrt(8e3 * self.params.R * T / (ca.pi * M_i))

    def init(self, r):
        self.D_co2_h2 = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_co2, self.params.M_h2,
                                       self.params.delta_v_co2, self.params.delta_v_h2)
        self.D_co2_ch4 = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_co2, self.params.M_ch4,
                                        self.params.delta_v_co2, self.params.delta_v_ch4)
        self.D_co2_h2o = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_co2, self.params.M_h2o,
                                        self.params.delta_v_co2, self.params.delta_v_h2o)
        self.D_h2_h2o = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_h2, self.params.M_h2o,
                                       self.params.delta_v_h2, self.params.delta_v_h2o)
        self.D_h2_ch4 = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_h2, self.params.M_ch4,
                                       self.params.delta_v_h2, self.params.delta_v_ch4)
        self.D_ch4_h2o = self.get_D_i_j(self.ctx.T[r], self.ctx.p[r], self.params.M_ch4, self.params.M_h2o,
                                        self.params.delta_v_ch4, self.params.delta_v_h2o)

    def get_D_co2(self, r):
        return ((self.params.epsilon / self.params.tau_sq
                 * (1 / self.get_D_i_m(self.ctx.y_co2[r], self.ctx.y_h2[r], self.ctx.y_ch4[r],
                                       self.ctx.y_h2o[r], self.D_co2_h2, self.D_co2_ch4, self.D_co2_h2o)
                    + 1e-6 / self.get_D_i_Kn(self.ctx.T[r], self.params.M_co2)) ** -1))

    def get_D_h2(self, r):
        return ((self.params.epsilon / self.params.tau_sq
                 * (1 / self.get_D_i_m(self.ctx.y_h2[r], self.ctx.y_co2[r], self.ctx.y_ch4[r],
                                       self.ctx.y_h2o[r], self.D_co2_h2, self.D_h2_ch4, self.D_h2_h2o)
                    + 1e-6 / self.get_D_i_Kn(self.ctx.T[r], self.params.M_h2)) ** -1))

    def get_D_ch4(self, r):
        return ((self.params.epsilon / self.params.tau_sq
                 * (1 / self.get_D_i_m(self.ctx.y_ch4[r], self.ctx.y_h2[r], self.ctx.y_co2[r],
                                       self.ctx.y_h2o[r], self.D_h2_ch4, self.D_co2_ch4, self.D_ch4_h2o)
                    + 1e-6 / self.get_D_i_Kn(self.ctx.T[r], self.params.M_ch4)) ** -1))
