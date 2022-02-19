# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:00:30 2022

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt

def bcm(w, theta, xi, y, epsilon=1):
    return y * (y - theta) * xi - epsilon

def sigmoid(x):
    return 1/(1 + np.exp(-x))
xi = 10
theta = 25
y = np.arange(0, 40)
dt = 0.0001

output1 = bcm(10, xi, y) * dt
output2 = bcm(20, xi, y) * dt
output3 = bcm(30, xi, y) * dt

plt.figure()
plt.plot(y, output1)
plt.plot(y, output2)
plt.plot(y, output3)
plt.hlines([0], xmin=0, xmax=40, color='k')

plt.legend(("theta=10", "theta=20", "theta=30"))
plt.xlabel("Postsynaptic activity")
plt.ylabel("Weight Change")