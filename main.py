from integrator_sin import IntegratorSin
from integrator_step import IntegratorStep
from parameters import Parameters

params = Parameters()

integrator = IntegratorSin(params)
integrator.run()

integrator = IntegratorStep(params)
integrator.run()
