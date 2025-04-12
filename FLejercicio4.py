import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# === 1. Universo y variables ===
left_sensor = ctrl.Antecedent(np.arange(0, 11, 1), 'left_sensor')
front_sensor = ctrl.Antecedent(np.arange(0, 11, 1), 'front_sensor')
right_sensor = ctrl.Antecedent(np.arange(0, 11, 1), 'right_sensor')

left_motor = ctrl.Consequent(np.arange(0, 11, 1), 'left_motor')
right_motor = ctrl.Consequent(np.arange(0, 11, 1), 'right_motor')

# === 2. Funciones de membresía solapadas ===
for sensor in [left_sensor, front_sensor, right_sensor]:
    sensor['near'] = fuzz.trapmf(sensor.universe, [0, 0, 3, 6])
    sensor['far'] = fuzz.trapmf(sensor.universe, [4, 7, 10, 10])

for motor in [left_motor, right_motor]:
    motor['forward'] = fuzz.trimf(motor.universe, [0, 3, 6])
    motor['reverse'] = fuzz.trimf(motor.universe, [4, 7, 10])

# === 3. Reglas difusas ===
rules = [
    ctrl.Rule(left_sensor['far'] & front_sensor['far'] & right_sensor['far'],
              (left_motor['forward'], right_motor['forward'])),
    ctrl.Rule(left_sensor['far'] & front_sensor['far'] & right_sensor['near'],
              (left_motor['forward'], right_motor['reverse'])),
    ctrl.Rule(left_sensor['far'] & front_sensor['near'] & right_sensor['far'],
              (left_motor['reverse'], right_motor['reverse'])),
    ctrl.Rule(left_sensor['far'] & front_sensor['near'] & right_sensor['near'],
              (left_motor['reverse'], right_motor['reverse'])),
    ctrl.Rule(left_sensor['near'] & front_sensor['far'] & right_sensor['far'],
              (left_motor['reverse'], right_motor['forward'])),
    ctrl.Rule(left_sensor['near'] & front_sensor['far'] & right_sensor['near'],
              (left_motor['reverse'], right_motor['reverse'])),
    ctrl.Rule(left_sensor['near'] & front_sensor['near'] & right_sensor['far'],
              (left_motor['reverse'], right_motor['reverse'])),
    ctrl.Rule(left_sensor['near'] & front_sensor['near'] & right_sensor['near'],
              (left_motor['reverse'], right_motor['reverse'])),
]

# === 4. Sistema de control ===
obstacle_avoidance_ctrl = ctrl.ControlSystem(rules)
obstacle_avoidance = ctrl.ControlSystemSimulation(obstacle_avoidance_ctrl)

# === 5. Prueba con entradas ===
obstacle_avoidance.input['left_sensor'] = 2
obstacle_avoidance.input['front_sensor'] = 5
obstacle_avoidance.input['right_sensor'] = 8

obstacle_avoidance.compute()

# === 6. Resultados ===
print("Left Motor Output:", obstacle_avoidance.output['left_motor'])
print("Right Motor Output:", obstacle_avoidance.output['right_motor'])

# === 7. Visualizar funciones de salida ===
left_motor.view(sim=obstacle_avoidance)
right_motor.view(sim=obstacle_avoidance)

# Visualizar entradas
left_sensor.view()
front_sensor.view()
right_sensor.view()

plt.show()
