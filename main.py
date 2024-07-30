from integrator import Integrator
from parameters import Parameters
import os

# check if plots dir exists
if not os.path.exists('plots'):
    os.mkdir('plots')

params = Parameters()

integrator = Integrator(params)
integrator.run()

# Betrags funktion casadi keine negativen werte -> funktioniert nicht
# TODO: massen strom am surface berechnen
