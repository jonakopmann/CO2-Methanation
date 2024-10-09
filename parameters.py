import casadi as ca
import numpy as np


class Parameters:
    # start mass fractions
    w_ar_0 = 0
    w_h2_0 = 0.15
    w_co2_0 = 0.85
    w_ch4_0 = 0
    w_h2o_0 = 1 - w_h2_0 - w_co2_0 - w_ch4_0 - w_ar_0

    # stoichiometric factors
    v_co2 = -1
    v_h2 = -4
    v_ch4 = 1
    v_h2o = 2

    # diffusion volumes from Fuller et al. 1969
    delta_v_co2 = 26.7  # [cm^3]
    delta_v_h2 = 6.12  # [cm^3]
    delta_v_ch4 = 25.14  # [cm^3]
    delta_v_h2o = 13.1  # [cm^3]

    # molar masses
    M_co2 = 44.0095  # [g/mol]
    M_h2 = 2.01588  # [g/mol]
    M_ch4 = 16.0425  # [g/mol]
    M_h2o = 18.0153  # [g/mol]
    M_0 = (w_co2_0 / M_co2 + w_h2_0 / M_h2 + w_ch4_0 / M_ch4 + w_h2o_0 / M_h2o) ** -1  # [g/mol]

    y_co2_0 = w_co2_0 * M_0 / M_co2
    y_h2_0 = w_h2_0 * M_0 / M_h2
    y_ch4_0 = w_ch4_0 * M_0 / M_ch4

    # dynamic const
    delta_y = 0.05
    delta_T = 5
    f_y = 1  # [1/s]
    f_T = 0.1  # [1/s]

    # constants
    p_0 = 8  # [bar]
    cp_s = 1107e-3  # [J/(K*g)] (nonporous)
    rho_s = 2355.2e3  # [g/m^3] (porous)
    epsilon = 0.6
    tau_sq = 4
    lambda_eff = 3.6e-3  # [W/(mm*K)]
    n = 2
    R = 8.314463  # [J/(mol*K)]
    d_pore = 10e-9  # [m]

    # reference values
    T_ref = 555  # [K]
    k_0_ref = 3.46e-4  # [mol/(bar*s*g)]
    EA = 77.5e3  # [J/mol]
    K_oh_ref = 0.5  # [1/bar^0.5]
    delta_H_oh = 22.4e3  # [J/mol]
    K_h2_ref = 0.44  # [1/bar^0.5]
    delta_H_h2 = -6.2e3  # [J/mol]
    K_mix_ref = 0.88  # [1/bar^0.5]
    delta_H_mix = -10e3  # [J/mol]

    # integration params
    r_steps = 100
    r_max = 1  # [mm]
    h = r_max / r_steps  # [mm]
    T_0 = 533   # [K]

    # conversion stuff
    T_max = 900  # [K]
    T_step = 5  # [K]

    # time stuff
    fps = 30  # [1/s]
    t_max = 20
    t_steps = fps * t_max
    t_i = np.linspace(0, t_max, t_steps)  # [s]
    x_min = 100

    # feed speed
    v = 1000  # [mm/s]
    factor = 0.32  # r_f = factor * r_max

    # start conditions
    x0 = ca.vertcat(np.full(r_steps, w_co2_0), np.full(r_steps, w_ch4_0), np.full(r_steps, w_h2_0),
                    np.full(r_steps, T_0))
    z0 = ca.vertcat(w_co2_0, w_ch4_0, w_h2_0, T_0, w_co2_0, w_ch4_0, w_h2_0, T_0)

    def refresh(self):
        self.x0 = ca.vertcat(np.full(self.r_steps, self.w_co2_0), np.full(self.r_steps, self.w_ch4_0), np.full(self.r_steps, self.w_h2_0),
                        np.full(self.r_steps, self.T_0))
        self.z0 = ca.vertcat(self.w_co2_0, self.w_ch4_0, self.w_h2_0, self.T_0, self.w_co2_0, self.w_ch4_0, self.w_h2_0, self.T_0)
