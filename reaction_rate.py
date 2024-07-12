import casadi as ca

from parameters import Parameters


def get_H_co2(T):
    t = T / 1000
    A = 24.99735
    B = 55.18696
    C = -33.69137
    D = 7.948387
    E = -0.136638
    F = -403.6075
    H = -393.5224
    return A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H


def get_H_h2(T):
    t = T / 1000
    A = 33.066178
    B = -11.363417
    C = 11.432816
    D = -2.772874
    E = -0.158558
    F = -9.980797
    H = 0.0
    return A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H


def get_H_ch4(T):
    t = T / 1000
    A = -0.703029
    B = 108.4773
    C = -42.52157
    D = 5.862788
    E = 0.678565
    F = -76.84376
    H = -74.87310
    return A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H


def get_H_h20(T):
    t = T / 1000
    A = 30.09200
    B = 6.832514
    C = 6.793435
    D = -2.534480
    E = 0.082139
    F = -250.8810
    H = -241.8264
    return A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H


def get_S_co2(T):
    t = (T - 298.15) / 1000
    A = 33.066178
    B = -11.363417
    C = 11.432816
    D = -2.772874
    E = -0.158558
    G = 172.707974
    return A * ca.log(t) + B * t + (C / 2) * t ** 2 + (D / 3) * t ** 3 - E / (2 * t ** 2)


def get_S_h2(T):
    t = (T - 298.15) / 1000
    A = 24.99735
    B = 55.18696
    C = -33.69137
    D = 7.948387
    E = -0.136638
    G = 228.2431
    return A * ca.log(t) + B * t + (C / 2) * t ** 2 + (D / 3) * t ** 3 - E / (2 * t ** 2)


def get_S_ch4(T):
    t = (T - 298.15) / 1000
    A = -0.703029
    B = 108.4773
    C = -42.5215
    D = 5.862788
    E = 0.678565
    G = 158.7163
    return A * ca.log(t) + B * t + (C / 2) * t ** 2 + (D / 3) * t ** 3 - E / (2 * t ** 2)


def get_S_h20(T):
    t = (T - 298.15) / 1000
    A = 30.09200
    B = 6.832514
    C = 6.793435
    D = -2.534480
    E = 0.082139
    G = 223.3967
    return A * ca.log(t) + B * t + (C / 2) * t ** 2 + (D / 3) * t ** 3 - E / (2 * t ** 2)


class ReactionRate:
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
        H_f_h2o = -241.8264 + get_H_h20(T)
        H_f_ch4 = -74.87310 + get_H_ch4(T)
        H_f_h2 = 0 + get_H_h2(T)
        H_f_co2 = -393.5224 + get_H_co2(T)

        return 1000 * (self.params.v_co2 * H_f_co2 + self.params.v_h2 * H_f_h2
                       + self.params.v_ch4 * H_f_ch4 + self.params.v_h2o * H_f_h2o)  # [J/mol]

    def get_K_eq(self, T):
        # exp(-delta_R_G/RT)
        S = (self.params.v_co2 * (213.79 + get_S_co2(T)) + self.params.v_h2 * (130.68 + get_S_h2(T))
             + self.params.v_ch4 * (186.25 + get_S_ch4(T)) + self.params.v_h2o * (188.84 + get_S_h20(T)))  # [J/(mol*K)]
        G = self.get_H_R(T) - T * S  # [J/mol]
        return ca.exp(-G / (self.params.R * T)) * self.params.p_t ** -2

    def calc(self, y_co2, y_h2, y_ch4, y_h2o, T):
        p_co2 = self.params.p_t * y_co2
        p_h2 = self.params.p_t * y_h2
        p_ch4 = self.params.p_t * y_ch4
        p_h2o = self.params.p_t * y_h2o
        return (self.get_k(T) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(T)))
                / ((1 + self.get_K_oh(T) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(T) *
                    (p_h2 ** 0.5) + self.get_K_mix(T) * (p_co2 ** 0.5)) ** 2))
