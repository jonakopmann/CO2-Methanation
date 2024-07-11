import numpy as np


class Parameters:
    # start mole fractions
    y_h2_0 = 0.7
    y_co2_0 = 0.1
    y_ch4_0 = 0
    y_h2o_0 = 0

    # stoichiometric factors
    v_h2 = -4
    v_co2 = -1
    v_ch4 = 1
    v_h2o = 2

    # dynamic const
    delta_y = 0.05
    delta_T = 0
    f = 1  # [1/s]

    # constants
    p_t = 1  # [bar]
    c_p = 880 * 1 / 1000  # [J/(K*g)]
    roh_s = 2350 * 1000  # [g/m^3]
    epsilon = 0.5
    tau = 3
    D_i_eff = 1  # [mm^2/s]
    lambda_eff = 0.67 / 1000  # [W/(mm*K)]
    n = 2
    R = 8.314463  # [J/(mol*K)]

    # reference values
    T_ref = 555  # [K]
    k_0_ref = 3.46e-4  # [mol/(bar*s*g)]
    EA = 77.5 * 1000  # [J/mol]
    K_oh_ref = 0.5  # [1/bar^0.5]
    delta_H_oh = 22.4 * 1000  # [J/mol]
    K_h2_ref = 0.44  # [1/bar^0.5]
    delta_H_h2 = -6.2 * 1000  # [J/mol]
    K_mix_ref = 0.88  # [1/bar^0.5]
    delta_H_mix = -10 * 1000  # [J/mol]
    K_eq_ref = 0

    # integration params
    r_steps = 100
    r_max = 1.5  # [mm]
    h = r_max / r_steps  # [mm]
    T_0 = 555  # [K]
    t_steps = 100
    t_i = np.linspace(0, 1, t_steps)  # [s]
