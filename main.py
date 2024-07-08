from integrator import Integrator
from parameters import Parameters

import casadi as ca

params = Parameters()

integrator = Integrator(params)
integrator.run()
