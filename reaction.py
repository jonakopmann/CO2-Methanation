from context import Context
from thermo import *


class Reaction:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.params = ctx.params

    def get_k(self, T):
        # k_0_ref * exp(E_A/R * (1/T_ref - 1/T))
        return self.params.k_0_ref * ca.exp(self.params.EA / self.params.R * ((1 / self.params.T_ref) - (1 / T)))

    def get_K_oh(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_oh_ref
                * ca.exp(self.params.delta_H_oh / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_K_h2(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_h2_ref
                * ca.exp(self.params.delta_H_h2 / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_K_mix(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_mix_ref
                * ca.exp(self.params.delta_H_mix / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_H_R(self, T):
        H_f_h2o = -241.8264 + get_H_h2o(T)
        H_f_ch4 = -74.87310 + get_H_ch4(T)
        H_f_h2 = 0 + get_H_h2(T)
        H_f_co2 = -393.5224 + get_H_co2(T)

        return 1e3 * (self.params.v_co2 * H_f_co2 + self.params.v_h2 * H_f_h2
                      + self.params.v_ch4 * H_f_ch4 + self.params.v_h2o * H_f_h2o)  # [J/mol]

    def get_K_eq(self, T, p):
        # exp(-delta_R_G/RT)
        S = (self.params.v_co2 * (get_S_co2(T)) + self.params.v_h2 * (get_S_h2(T))
             + self.params.v_ch4 * (get_S_ch4(T)) + self.params.v_h2o * (get_S_h2o(T)))  # [J/(mol*K)]
        G = self.get_H_R(T) - T * S  # [J/mol]
        return ca.exp(-G / (self.params.R * T)) * p ** -2

    def get_K_eq_imp(self, T, p):
        return 137 * (T ** -3.998) * ca.exp(158.7e3 / (self.params.R * T))

    def get_p_i(self, w_i, M_i, i):
        return self.ctx.p[i] * w_i * self.ctx.M[i] / M_i

    def get_r(self, i):
        p_co2 = self.get_p_i(self.ctx.w_co2[i], self.params.M_co2, i)
        p_h2 = self.get_p_i(self.ctx.w_h2[i], self.params.M_h2, i)
        p_ch4 = self.get_p_i(self.ctx.w_ch4[i], self.params.M_ch4, i)
        p_h2o = self.get_p_i(self.ctx.w_h2o[i], self.params.M_h2o, i)

        r = (self.get_k(self.ctx.T[i]) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(self.ctx.T[i], self.ctx.p[i])))
             / ((1 + self.get_K_oh(self.ctx.T[i]) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(self.ctx.T[i]) *
                 (p_h2 ** 0.5) + self.get_K_mix(self.ctx.T[i]) * (p_co2 ** 0.5)) ** 2))

        return ca.if_else(self.ctx.w_co2[i] < 1e-20, 1e-30, r)

    def get_mass_term(self, M_i, rho_g, v_i, r):
        return ((1 - self.params.epsilon) / self.params.epsilon) * M_i / rho_g * self.params.rho_s * v_i * r

    def get_heat_term(self, i, r):
        return (-self.get_H_R(self.ctx.T[i]) * (1 - self.params.epsilon) * self.params.rho_s
                / ((1 - self.params.epsilon) * self.params.rho_s * self.params.cp_s + self.params.epsilon * self.ctx.rho[i] * self.ctx.cp[i]) * r)
