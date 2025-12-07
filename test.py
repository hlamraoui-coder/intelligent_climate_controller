import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib.patches import Wedge

# -----------------------------
# Fuzzy variables
# -----------------------------
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
heating = ctrl.Consequent(np.arange(0, 101, 1), 'heating')

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['comfortable'] = fuzz.trimf(temperature.universe, [15, 20, 25])
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 40, 40])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['comfortable'] = fuzz.trimf(humidity.universe, [40, 50, 60])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

heating['off'] = fuzz.trimf(heating.universe, [0, 0, 50])
heating['low'] = fuzz.trimf(heating.universe, [0, 50, 100])
heating['high'] = fuzz.trimf(heating.universe, [50, 100, 100])

rules = [
    ctrl.Rule(temperature['cold'] & humidity['low'], heating['high']),
    ctrl.Rule(temperature['cold'] & humidity['comfortable'], heating['low']),
    ctrl.Rule(temperature['comfortable'] & humidity['low'], heating['low']),
    ctrl.Rule(temperature['comfortable'] & humidity['comfortable'], heating['off']),
    ctrl.Rule(temperature['hot'] & humidity['high'], heating['off']),
    ctrl.Rule(temperature['hot'] & humidity['comfortable'], heating['low']),
    ctrl.Rule(temperature['hot'] & humidity['low'], heating['high']),
]

heating_ctrl = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(heating_ctrl)

# -----------------------------
# Figure setup
# -----------------------------
fig, (ax_dial, ax_slice) = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Dial axis
ax_dial.set_xlim(-1.1, 1.1)
ax_dial.set_ylim(-1.1, 1.1)
ax_dial.set_aspect('equal')
ax_dial.axis('off')
circle = plt.Circle((0,0), 1, color='lightgrey', alpha=0.3)
ax_dial.add_artist(circle)
wedge = Wedge((0,0), 1, 0, 0, width=0.2, color='red', alpha=0.7)
ax_dial.add_artist(wedge)

# Slice axis for decision surface
ax_slice.set_xlim(0, 40)
ax_slice.set_ylim(0, 100)
ax_slice.set_xlabel("Temperature (°C)")
ax_slice.set_ylabel("Heating Output (%)")
ax_slice.set_title("Decision Surface Slice")
line_slice, = ax_slice.plot([], [], lw=3, color='blue')

# -----------------------------
# Input widgets
# -----------------------------
temp_ax = plt.axes([0.05, 0.15, 0.1, 0.05])
temp_box = TextBox(temp_ax, 'Temp', initial="20")

hum_ax = plt.axes([0.05, 0.08, 0.1, 0.05])
hum_box = TextBox(hum_ax, 'Humidity', initial="50")

button_ax = plt.axes([0.8, 0.1, 0.1, 0.075])
button = Button(button_ax, 'Compute')

output_ax = plt.axes([0.25, 0.05, 0.5, 0.05])
output_text = TextBox(output_ax, 'Heating Output (%)', initial="0")

# Colormap
cmap = plt.cm.get_cmap('coolwarm')

# -----------------------------
# Compute and update function
# -----------------------------
def update(event):
    try:
        t_val = float(temp_box.text)
        h_val = float(hum_box.text)
    except:
        output_text.set_val("Invalid input!")
        return
    
    # Fuzzy logic simulation
    sim.input['temperature'] = t_val
    sim.input['humidity'] = h_val
    sim.compute()
    heat_val = sim.output['heating']
    
    # Update dial
    wedge.set_theta1(0)
    wedge.set_theta2((heat_val/100)*360)
    wedge.set_facecolor(cmap(heat_val/100))
    
    # Update output
    output_text.set_val(f"{heat_val:.2f}")
    
    # Decision surface slice (temperature vs heating)
    temp_range = np.arange(0, 41, 1)
    heating_slice = []
    for temp in temp_range:
        sim.input['temperature'] = temp
        sim.input['humidity'] = h_val
        sim.compute()
        heating_slice.append(sim.output['heating'])
    
    line_slice.set_data(temp_range, heating_slice)
    ax_slice.relim()
    ax_slice.autoscale_view()
    
    fig.canvas.draw_idle()

button.on_clicked(update)

plt.show()
