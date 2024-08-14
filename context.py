from parameters import Parameters
from thermo import *


class Context:
    def __init__(self, params: Parameters):
        self.params = params

        self.t = ca.SX.sym('t')
        self.p = ca.SX.sym('p', self.params.r_steps)

        self.w_co2 = ca.SX.sym('w_co2', self.params.r_steps)
        self.w_ch4 = ca.SX.sym('w_ch4', self.params.r_steps)
        self.w_h2 = ca.SX.sym('w_h2', self.params.r_steps)
        self.w_h2o = 1 - self.w_co2 - self.w_ch4 - self.w_h2
        self.T = ca.SX.sym('T', self.params.r_steps)
        self.M = (self.w_co2 / self.params.M_co2 + self.w_h2 / self.params.M_h2
                  + self.w_ch4 / self.params.M_ch4 + self.w_h2o / self.params.M_h2o) ** -1
        self.rho = self.p * 1e5 * self.M / (self.params.R * self.T)

        # mole fraction
        self.y_co2 = w_to_y(self.w_co2, self.params.M_co2, self.M)
        self.y_h2 = w_to_y(self.w_h2, self.params.M_h2, self.M)
        self.y_ch4 = w_to_y(self.w_ch4, self.params.M_ch4, self.M)
        self.y_h2o = w_to_y(self.w_h2o, self.params.M_h2o, self.M)

        # surface
        self.w_co2_surf = ca.SX.sym('w_co2_surf')
        self.w_h2_surf = ca.SX.sym('w_h2_surf')
        self.w_ch4_surf = ca.SX.sym('w_ch4_surf')
        self.w_h2o_surf = 1 - self.w_co2_surf - self.w_h2_surf - self.w_ch4_surf
        self.T_surf = ca.SX.sym('T_surf')
        self.M_surf = (self.w_co2_surf / self.params.M_co2 + self.w_h2_surf / self.params.M_h2
                       + self.w_ch4_surf / self.params.M_ch4 + self.w_h2o_surf / self.params.M_h2o) ** -1
        self.rho_surf = self.p[-1] * 1e5 * self.M_surf / (self.params.R * self.T_surf)

        # fluid
        self.w_co2_fl = ca.SX.sym('w_co2_fl')
        self.w_h2_fl = ca.SX.sym('w_h2_fl')
        self.w_ch4_fl = ca.SX.sym('w_ch4_fl')
        self.w_h2o_fl = 1 - self.w_co2_fl - self.w_h2_fl - self.w_ch4_fl
        self.T_fl = ca.SX.sym('T_fl')
        self.M_fl = (self.w_co2_fl / self.params.M_co2 + self.w_h2_fl / self.params.M_h2
                     + self.w_ch4_fl / self.params.M_ch4 + self.w_h2o_fl / self.params.M_h2o) ** -1
        self.rho_fl = self.p[-1] * 1e5 * self.M_fl / (self.params.R * self.T_fl)

        # diffusion
        self.D_co2_eff = ca.SX.sym('D_co2_eff', self.params.r_steps)
        self.D_ch4_eff = ca.SX.sym('D_ch4_eff', self.params.r_steps)
        self.D_h2_eff = ca.SX.sym('D_h2_eff', self.params.r_steps)

        # heat transfer
        cp_fl = (self.w_co2_fl * get_cp_co2(self.T_fl) + self.w_h2_fl * get_cp_h2(self.T_fl)
                 + self.w_ch4_fl * get_cp_ch4(self.T_fl) + self.w_h2o_fl * get_cp_h2o(self.T_fl))  # [J/(g*K)]

        ny_fl = (self.w_co2_fl * get_ny_co2(self.T_fl, self.p[-1]) + self.w_h2_fl * get_ny_h2(self.T_fl, self.p[-1])
                 + self.w_ch4_fl * get_ny_ch4(self.T_fl, self.p[-1]) + self.w_h2o_fl * get_ny_h2o(self.T_fl, self.p[-1]))  # [mm^2/s]

        lambda_fl = (self.w_co2_fl * get_lambda_co2(self.T_fl) + self.w_h2_fl * get_lambda_h2(self.T_fl)
                     + self.w_ch4_fl * get_lambda_ch4(self.T_fl) + self.w_h2o_fl * get_lambda_h2o(self.T_fl))  # [W/(mm*K)]

        Re = self.params.v * self.params.r_max * 2 / ny_fl
        Pr = ny_fl / lambda_fl * cp_fl * self.rho_fl * 1e-9
        Nu = 2 + 0.6 * Re ** 0.5 + Pr ** (1 / 3)
        self.alpha = Nu * lambda_fl / (2 * self.params.r_max)  # [W/(mm^2*K)]

        # species transfer
        Sc_co2 = ny_fl / self.D_co2_eff[-1]
        Sh_co2 = 2 + 0.6 * Re ** 0.5 + Sc_co2 ** (1 / 3)
        self.beta_co2 = Sh_co2 * self.D_co2_eff[-1] / (2 * self.params.r_max)  # [mm/s]

        Sc_h2 = ny_fl / self.D_h2_eff[-1]
        Sh_h2 = 2 + 0.6 * Re ** 0.5 + Sc_h2 ** (1 / 3)
        self.beta_h2 = Sh_h2 * self.D_h2_eff[-1] / (2 * self.params.r_max)  # [mm/s]

        Sc_ch4 = ny_fl / self.D_ch4_eff[-1]
        Sh_ch4 = 2 + 0.6 * Re ** 0.5 + Sc_ch4 ** (1 / 3)
        self.beta_ch4 = Sh_ch4 * self.D_ch4_eff[-1] / (2 * self.params.r_max)  # [mm/s]
