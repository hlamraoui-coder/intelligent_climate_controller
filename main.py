import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib.patches import Wedge

# -----------------------------
# Define fuzzy variables
# -----------------------------
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
heating = ctrl.Consequent(np.arange(0, 101, 1), 'heating')

# Membership functions
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['comfortable'] = fuzz.trimf(temperature.universe, [15, 20, 25])
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 40, 40])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['comfortable'] = fuzz.trimf(humidity.universe, [40, 50, 60])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

heating['off'] = fuzz.trimf(heating.universe, [0, 0, 50])
heating['low'] = fuzz.trimf(heating.universe, [0, 50, 100])
heating['high'] = fuzz.trimf(heating.universe, [50, 100, 100])

# -----------------------------
# Define fuzzy rules
# -----------------------------
rules = [
    ctrl.Rule(temperature['cold'] & humidity['low'], heating['high']),
    ctrl.Rule(temperature['cold'] & humidity['comfortable'], heating['low']),
    ctrl.Rule(temperature['comfortable'] & humidity['low'], heating['low']),
    ctrl.Rule(temperature['comfortable'] & humidity['comfortable'], heating['off']),
    ctrl.Rule(temperature['hot'] & humidity['high'], heating['off']),
    ctrl.Rule(temperature['hot'] & humidity['comfortable'], heating['low']),
    ctrl.Rule(temperature['hot'] & humidity['low'], heating['high'])
]

heating_ctrl = ctrl.ControlSystem(rules)

# -----------------------------
# Plot membership functions (non-blocking)
# -----------------------------
temperature.view()
plt.show(block=False)
plt.pause(0.5)

humidity.view()
plt.show(block=False)
plt.pause(0.5)

heating.view()
plt.show(block=False)
plt.pause(0.5)

# -----------------------------
# Decision surface (non-blocking)
# -----------------------------
temp_range = np.linspace(0, 40, 41)
hum_range = np.linspace(0, 100, 101)
heating_output_surface = np.zeros((len(temp_range), len(hum_range)))

for i, temp_val in enumerate(temp_range):
    for j, hum_val in enumerate(hum_range):
        sim = ctrl.ControlSystemSimulation(heating_ctrl)
        sim.input['temperature'] = temp_val
        sim.input['humidity'] = hum_val
        try:
            sim.compute()
            heating_output_surface[i, j] = sim.output['heating']
        except:
            heating_output_surface[i, j] = 0

plt.figure()
plt.imshow(heating_output_surface, extent=(0, 100, 0, 40), origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='Heating Output (%)')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Decision Surface for Heating Control')
plt.show(block=False)
plt.pause(0.5)

# -----------------------------
# Test multiple real cases
# -----------------------------
test_cases = [
    (10, 30),
    (20, 50),
    (30, 70),
    (15, 40),
    (35, 20)
]

print("\n--- Test Cases ---")
for temp_val, hum_val in test_cases:
    sim = ctrl.ControlSystemSimulation(heating_ctrl)
    sim.input['temperature'] = temp_val
    sim.input['humidity'] = hum_val
    try:
        sim.compute()
        output = sim.output['heating']
    except:
        output = 0
    print(f"Temperature: {temp_val}°C, Humidity: {hum_val}%, Heating Output: {output:.2f}%")

# -----------------------------
# Interactive Circular Heater Dial
# -----------------------------
current_temp = 20
current_hum = 50

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')

# Draw dial background
circle = plt.Circle((0,0), 1, color='lightgrey', alpha=0.3)
ax.add_artist(circle)

# Slider axes
ax_temp = plt.axes([0.1, 0.1, 0.8, 0.03])
ax_hum = plt.axes([0.1, 0.05, 0.8, 0.03])
slider_temp = Slider(ax_temp, 'Temperature', 0, 40, valinit=current_temp)
slider_hum = Slider(ax_hum, 'Humidity', 0, 100, valinit=current_hum)

# Heating wedge (moving indicator)
wedge = Wedge((0,0), 1, 0, 0, width=0.2, color='red', alpha=0.7)
ax.add_artist(wedge)

# Colormap for color change
cmap = plt.cm.get_cmap('coolwarm')

def update(val):
    t = slider_temp.val
    h = slider_hum.val
    sim = ctrl.ControlSystemSimulation(heating_ctrl)
    sim.input['temperature'] = t
    sim.input['humidity'] = h
    try:
        sim.compute()
        heat_val = sim.output['heating']
    except:
        heat_val = 0
    angle = (heat_val / 100) * 360
    wedge.set_theta1(0)
    wedge.set_theta2(angle)
    wedge.set_facecolor(cmap(heat_val / 100))
    fig.canvas.draw_idle()

slider_temp.on_changed(update)
slider_hum.on_changed(update)
update(None)

plt.title("Dynamic Circular Heater Dial")
plt.show()  # block=True ensures dial stays open
