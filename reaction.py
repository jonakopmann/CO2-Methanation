from parameters import Parameters
from thermo import *


class Reaction:
    def __init__(self, params: Parameters):
        self.params = params

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

    def get_r(self, w_co2, w_h2, w_ch4, w_h2o, T, p, M):
        p_co2 = p * w_co2 * M / self.params.M_co2
        p_h2 = p * w_h2 * M / self.params.M_h2
        p_ch4 = p * w_ch4 * M / self.params.M_ch4
        p_h2o = p * w_h2o * M / self.params.M_h2o

        a = (self.get_k(T) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(T, p)))
             / ((1 + self.get_K_oh(T) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(T) *
                 (p_h2 ** 0.5) + self.get_K_mix(T) * (p_co2 ** 0.5)) ** 2))

        return ca.if_else(w_co2 < 1e-20, 1e-20, a)

    def get_mass_term(self, M_i, roh_g, v_i, r):
        return ((1 - self.params.epsilon) / self.params.epsilon) * M_i / roh_g * self.params.roh_s * v_i * r

    def get_heat_term(self, T, r):
        return -self.get_H_R(T) / self.params.cp_s * (1 - self.params.epsilon) * r
