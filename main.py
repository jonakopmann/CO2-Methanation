from integrator import Integrator
from parameters import Parameters

params = Parameters()

integrator = Integrator(params)
integrator.run()

# Betrags funktion casadi keine negativen werte -> funktioniert nicht
# TODO: massen strom am surface berechnen
