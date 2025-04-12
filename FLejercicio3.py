import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# === 1. Definir variables del sistema ===

# Entradas
service = ctrl.Antecedent(np.arange(0, 11, 0.1), 'service')
food = ctrl.Antecedent(np.arange(0, 11, 0.1), 'food')

# Salida
tip = ctrl.Consequent(np.arange(0, 31, 0.1), 'tip')

# === 2. Funciones de membresía ===

# Entrada: Service
service['poor'] = fuzz.gaussmf(service.universe, 0.0, 1.5)
service['good'] = fuzz.gaussmf(service.universe, 5.0, 1.5)
service['excellent'] = fuzz.gaussmf(service.universe, 10.0, 1.5)

# Entrada: Food
food['rancid'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['delicious'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])

# Salida: Tip (gaussianas)
tip['low'] = fuzz.gaussmf(tip.universe, 0.0, 3.0)
tip['medium'] = fuzz.gaussmf(tip.universe, 12.5, 3.0)
tip['high'] = fuzz.gaussmf(tip.universe, 25.0, 3.0)

# === 3. Reglas difusas ===

rule1 = ctrl.Rule(service['poor'] | food['rancid'], tip['low'])
rule2 = ctrl.Rule(service['good'], tip['medium'])
rule3 = ctrl.Rule(service['excellent'] | food['delicious'], tip['high'])

# === 4. Crear el sistema ===

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# === 5. Prueba ===

tipping.input['service'] = 2.0
tipping.input['food'] = 7.0

tipping.compute()

# === 6. Mostrar resultados ===
print("y =", tipping.output['tip'])

# === 7. Graficar salida ===
tip.view(sim=tipping)
plt.show()
