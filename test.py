import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

# Define the time variable (dimensionless)
t = np.linspace(0, 2 * np.pi, 1000)  # 0 to 2π

# Define two different frequencies
frequency1 = 1  # Frequency 1
frequency2 = 2  # Frequency 2

# Create the sinusoidal functions
sinusoid1 = np.sin(frequency1 * t)
sinusoid2 = np.sin(frequency2 * t)

colors = plt.cm.Dark2(np.linspace(0, 1, 8))

# Plotting
plt.figure(figsize=(3.5, 3.5))
plt.plot(t, sinusoid1, color=colors[0])
plt.plot(t, sinusoid2, color=colors[1])
plt.