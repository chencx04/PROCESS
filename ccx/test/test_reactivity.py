import process.models.physics.fusion_reactions as fusion_reactions
import numpy as np
import matplotlib.pyplot as plt

"""
test the reactivity of Tentori-Belloni and Nevins
plot the reactivity vs temperature
"""
temp_ion = np.linspace(10, 500, 1000)
sigma_v_tentori_belloni = np.array([fusion_reactions.tentori_belloni_reactivity(temp) for temp in temp_ion])
sigma_v_nevins = np.array([fusion_reactions.nevins_reactivity(temp) for temp in temp_ion])
plt.plot(temp_ion, sigma_v_tentori_belloni, label="tentori_belloni")
plt.plot(temp_ion, sigma_v_nevins, label="nevins")
plt.xlabel("Temperature (keV)")
plt.ylabel("sigma_v")
plt.title("sigma_v of Tentori-Belloni and Nevins")
plt.legend()
plt.savefig("ccx/test/reactivity.png")