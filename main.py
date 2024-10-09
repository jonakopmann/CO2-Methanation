from integrator import Integrator
from integrator_sin import IntegratorSin
from integrator_step import IntegratorStep
from parameters import Parameters

params = Parameters()

# for i in range(params.T_0, params.T_max + params.T_step, params.T_step):
#     params.T_0 = i
#     params.refresh()
#     integrator = IntegratorSin(params)
#     integrator.run()
#
integrator = IntegratorSin(params)
integrator.run()
#
# integrator = IntegratorStep(params)
# integrator.run()
# #
# params.f_y = 0.1
# params.f_T = 1
# integrator = IntegratorSin(params)
# integrator.run()

# params.f_y = 0.1
# params.f_T = 1
# integrator = IntegratorSin(params)
# integrator.run()
#
# params.f_y = 1
# params.f_T = 0.1
# integrator = IntegratorSin(params)
# integrator.run()

# params.f_y = 2
# params.f_T = 0.1
# integrator = IntegratorSin(params)
# integrator.run()
#
# integrator = IntegratorStep(params)
# integrator.run()
