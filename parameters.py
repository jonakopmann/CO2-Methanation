import numpy as np


class Parameters:
    # start mole fractions
    w_h2_0 = 0.15
    w_co2_0 = 0.85
    w_ch4_0 = 0
    w_h2o_0 = 0
    w_ar = 1 - w_h2_0 - w_co2_0 - w_ch4_0 - w_h2o_0

    # stoichiometric factors
    v_h2 = -4
    v_co2 = -1
    v_ch4 = 1
    v_h2o = 2

    # diffusion volumes
    delta_v_h2 = 6.12  # [cm^3]
    delta_v_co2 = 26.7  # [cm^3]
    delta_v_ch4 = 25.14  # [cm^3] TODO: not found in perrys book
    delta_v_h2o = 13.1  # [cm^3]

    # molar masses
    M_co2 = 44.0095  # [g/mol]
    M_h2 = 2.01588  # [g/mol]
    M_ch4 = 16.0425  # [g/mol]
    M_h2o = 18.0153  # [g/mol]
    M_0 = (w_co2_0 / M_co2 + w_h2_0 / M_h2 + w_ch4_0 / M_ch4 + w_h2o_0 / M_h2o) ** -1  # [g/mol]

    # dynamic const
    delta_y = 0.05
    delta_T = 0.5
    f_y = 2  # [1/s]
    f_T = 0.7  # [1/s]

    # constants
    p_0 = 8  # [bar]
    cp_s = 880e-3  # [J/(K*g)]
    roh_s = 2350e3  # [g/m^3]
    epsilon = 0.5
    tau = 4
    D_i_eff = 1  # [mm^2/s]
    lambda_eff = 0.67e-3  # [W/(mm*K)]
    n = 2
    R = 8.314463  # [J/(mol*K)]
    d_pore = 15e-9  # [m]

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
    r_max = 1.5  # [mm]
    h = r_max / r_steps  # [mm]
    T_0 = 525  # [K]
    t_steps = 100
    t_max = 2
    t_i = np.linspace(0, t_max, t_steps)  # [s]

    # heat transfer
    roh_fl = 0.7359e3  # [g/m^3]
    cp_fl = 1.025  # [J/(g*K)]
    ny_fl = 35.39  # [mm^2/s]
    v = 100  # [mm/s]
    lambda_fl = 38.35e-3  # [W/(mm*K)]

    Re = v * r_max * 2 / (ny_fl * epsilon)
    Pr = ny_fl / lambda_fl * cp_fl * roh_fl * 1e-9
    Nu_lam = 0.664 * (Re ** 0.5) * (Pr ** (3/2))
    Nu_turb = 0.037 * (Re ** 0.8) * Pr / (1 + 2.443 * (Re ** -0.1) * (Pr ** (2/3) - 1))
    Nu = 2 + (Nu_lam ** 2 + Nu_turb ** 2) ** 0.5
    alpha = Nu * lambda_fl / (2 * r_max)  # [W/(mm^2*K)]
