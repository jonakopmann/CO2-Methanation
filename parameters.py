import numpy as np


class Parameters:
    # start mass fractions
    w_h2_0 = 0.15
    w_co2_0 = 0.85
    w_ch4_0 = 0
    w_h2o_0 = 0

    # stoichiometric factors
    v_h2 = -4
    v_co2 = -1
    v_ch4 = 1
    v_h2o = 2

    # diffusion volumes from Fuller et al. 1969
    delta_v_h2 = 6.12  # [cm^3]
    delta_v_co2 = 26.7  # [cm^3]
    delta_v_ch4 = 25.14  # [cm^3]
    delta_v_h2o = 13.1  # [cm^3]

    # molar masses
    M_co2 = 44.0095  # [g/mol]
    M_h2 = 2.01588  # [g/mol]
    M_ch4 = 16.0425  # [g/mol]
    M_h2o = 18.0153  # [g/mol]
    M_0 = (w_co2_0 / M_co2 + w_h2_0 / M_h2 + w_ch4_0 / M_ch4 + w_h2o_0 / M_h2o) ** -1  # [g/mol]

    # dynamic const
    delta_w = 0.05
    delta_T = 5
    f_w = 1  # [1/s]
    f_T = 0.7  # [1/s]

    # constants
    p_0 = 8  # [bar]
    cp_s = 880e-3  # [J/(K*g)]
    roh_s = 2350e3  # [g/m^3]
    epsilon = 0.5
    tau = 4
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
    t_steps = 200
    t_max = 20
    t_i = np.linspace(0, t_max, t_steps)  # [s]

    # heat transfer
    v = 1000  # [mm/s]
