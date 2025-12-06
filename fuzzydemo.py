import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -----------------------------
# Define fuzzy variables
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

# -----------------------------
# Define rules as a tuple list
# (temp_label, hum_label, heating_label)
# -----------------------------
demo_rules = [
    ('cold', 'low', 'high'),
    ('cold', 'comfortable', 'low'),
    ('comfortable', 'low', 'low'),
    ('comfortable', 'comfortable', 'off'),
    ('hot', 'high', 'off'),
    ('hot', 'comfortable', 'low'),
    ('hot', 'low', 'high')
]

# -----------------------------
# Setup figure
# -----------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
ax_temp_plot, ax_hum_plot, ax_rule_plot, ax_agg_plot = axs

colors = ['blue', 'green', 'red']

# Plot temperature membership functions
for i, label in enumerate(temperature.terms.keys()):
    ax_temp_plot.plot(temperature.universe, temperature[label].mf, color=colors[i], label=label)
ax_temp_plot.set_title("Temperature Memberships")
ax_temp_plot.set_ylim(0, 1.1)
ax_temp_plot.legend()

# Plot humidity membership functions
for i, label in enumerate(humidity.terms.keys()):
    ax_hum_plot.plot(humidity.universe, humidity[label].mf, color=colors[i], label=label)
ax_hum_plot.set_title("Humidity Memberships")
ax_hum_plot.set_ylim(0, 1.1)
ax_hum_plot.legend()

# Rule bars
rule_names = [
    "Cold & Low -> High", "Cold & Comfortable -> Low", "Comfortable & Low -> Low",
    "Comfortable & Comfortable -> Off", "Hot & High -> Off", "Hot & Comfortable -> Low",
    "Hot & Low -> High"
]
ax_rule_plot.set_xticks(range(len(rule_names)))
ax_rule_plot.set_xticklabels(rule_names, rotation=30, ha='right')
rule_bars = ax_rule_plot.bar(range(len(demo_rules)), [0]*len(demo_rules), color='purple')
ax_rule_plot.set_title("Rule Activation Strengths")
ax_rule_plot.set_ylim(0,1)

# Aggregated heating output
agg_line, = ax_agg_plot.plot(heating.universe, np.zeros_like(heating.universe), color='purple', linewidth=2)
needle, = ax_agg_plot.plot([0,0],[0,1.1], color='black', linewidth=3)
ax_agg_plot.set_title("Aggregated Heating Output")
ax_agg_plot.set_ylim(0,1.1)
ax_agg_plot.set_xlabel("Heating (%)")
ax_agg_plot.set_ylabel("Membership")

# Sliders
ax_temp_slider = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_hum_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider_temp = Slider(ax_temp_slider, 'Temperature', 0, 40, valinit=20)
slider_hum = Slider(ax_hum_slider, 'Humidity', 0, 100, valinit=50)

# Keep references to dots for safe removal
temp_dots = []
hum_dots = []

# -----------------------------
# Update function
# -----------------------------
def update(val):
    t_val = slider_temp.val
    h_val = slider_hum.val

    # Remove old dots
    for d in temp_dots: d.remove()
    temp_dots.clear()
    for d in hum_dots: d.remove()
    hum_dots.clear()

    # Membership dots
    for i, label in enumerate(temperature.terms.keys()):
        deg = fuzz.interp_membership(temperature.universe, temperature[label].mf, t_val)
        dot, = ax_temp_plot.plot(t_val, deg, 'o', color=colors[i], markersize=10)
        temp_dots.append(dot)
    for i, label in enumerate(humidity.terms.keys()):
        deg = fuzz.interp_membership(humidity.universe, humidity[label].mf, h_val)
        dot, = ax_hum_plot.plot(h_val, deg, 'o', color=colors[i], markersize=10)
        hum_dots.append(dot)

    # Rule firing strengths
    firing_strengths = []
    for temp_label, hum_label, heat_label in demo_rules:
        deg_temp = fuzz.interp_membership(temperature.universe, temperature[temp_label].mf, t_val)
        deg_hum = fuzz.interp_membership(humidity.universe, humidity[hum_label].mf, h_val)
        firing_strengths.append(np.min([deg_temp, deg_hum]))

    # Update rule bars
    for bar, fs in zip(rule_bars, firing_strengths):
        bar.set_height(fs)
        bar.set_color(plt.cm.plasma(fs))

    # Aggregate output
    agg = np.zeros_like(heating.universe)
    for fs, (_, _, heat_label) in zip(firing_strengths, demo_rules):
        agg = np.fmax(agg, np.fmin(fs, heating[heat_label].mf))

    agg_line.set_ydata(agg)
    centroid = np.sum(agg * heating.universe) / np.sum(agg) if np.sum(agg)!=0 else 0
    needle.set_xdata([centroid, centroid])

    fig.canvas.draw_idle()

slider_temp.on_changed(update)
slider_hum.on_changed(update)
update(None)

plt.show()
