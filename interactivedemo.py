import tkinter as tk
from tkinter import ttk
import numpy as np
import skfuzzy as fuzz

# -----------------------------
# Fuzzy sets
# -----------------------------
temp_range = np.arange(0, 41, 1)
humidity_range = np.arange(0, 101, 1)
heating_range = np.arange(0, 101, 1)

# Membership functions
temp_cold = fuzz.trimf(temp_range, [0, 0, 20])
temp_comf = fuzz.trimf(temp_range, [15, 20, 25])
temp_hot = fuzz.trimf(temp_range, [20, 40, 40])

hum_low = fuzz.trimf(humidity_range, [0, 0, 50])
hum_comf = fuzz.trimf(humidity_range, [40, 50, 60])
hum_high = fuzz.trimf(humidity_range, [50, 100, 100])

heat_off = fuzz.trimf(heating_range, [0, 0, 50])
heat_low = fuzz.trimf(heating_range, [0, 50, 100])
heat_high = fuzz.trimf(heating_range, [50, 100, 100])

# Fuzzy rules
rules = [
    {'temp': 'cold', 'hum': 'low', 'heat': 'high'},
    {'temp': 'cold', 'hum': 'comfortable', 'heat': 'low'},
    {'temp': 'comfortable', 'hum': 'low', 'heat': 'low'},
    {'temp': 'comfortable', 'hum': 'comfortable', 'heat': 'off'},
    {'temp': 'hot', 'hum': 'high', 'heat': 'off'},
    {'temp': 'hot', 'hum': 'comfortable', 'heat': 'low'},
    {'temp': 'hot', 'hum': 'low', 'heat': 'high'}
]

# -----------------------------
# Helper functions
# -----------------------------
def get_temp_membership(value):
    return {
        'cold': fuzz.interp_membership(temp_range, temp_cold, value),
        'comfortable': fuzz.interp_membership(temp_range, temp_comf, value),
        'hot': fuzz.interp_membership(temp_range, temp_hot, value)
    }

def get_hum_membership(value):
    return {
        'low': fuzz.interp_membership(humidity_range, hum_low, value),
        'comfortable': fuzz.interp_membership(humidity_range, hum_comf, value),
        'high': fuzz.interp_membership(humidity_range, hum_high, value)
    }

def get_heat_mf(name, alpha):
    if name == 'off':
        return np.fmin(alpha, heat_off)
    elif name == 'low':
        return np.fmin(alpha, heat_low)
    elif name == 'high':
        return np.fmin(alpha, heat_high)

# -----------------------------
# GUI functions
# -----------------------------
def compute_fuzzy():
    output_text.delete("1.0", tk.END)  # clear previous output
    
    temp_val = float(temp_entry.get())
    hum_val = float(hum_entry.get())
    
    output_text.insert(tk.END, f"Input Temperature: {temp_val}°C\n")
    output_text.insert(tk.END, f"Input Humidity: {hum_val}%\n\n")
    
    # Fuzzification
    temp_memberships = get_temp_membership(temp_val)
    hum_memberships = get_hum_membership(hum_val)
    
    output_text.insert(tk.END, "--- Fuzzification ---\n")
    for k,v in temp_memberships.items():
        output_text.insert(tk.END, f"Temp '{k}': {v:.2f}\n")
    for k,v in hum_memberships.items():
        output_text.insert(tk.END, f"Humidity '{k}': {v:.2f}\n")
    
    # Rule evaluation
    output_text.insert(tk.END, "\n--- Rule Activations ---\n")
    fired_strengths = []
    for i, rule in enumerate(rules):
        alpha_temp = temp_memberships[rule['temp']]
        alpha_hum = hum_memberships[rule['hum']]
        firing_strength = np.min([alpha_temp, alpha_hum])
        fired_strengths.append((rule['heat'], firing_strength))
        output_text.insert(tk.END, f"Rule {i+1}: IF Temp is {rule['temp']} AND Humidity is {rule['hum']} THEN Heating is {rule['heat']}, firing strength = {firing_strength:.2f}\n")
    
    # Aggregation
    agg_heat = np.zeros_like(heating_range, dtype=float)
    for heat_type, alpha in fired_strengths:
        agg_heat = np.fmax(agg_heat, get_heat_mf(heat_type, alpha))
    output_text.insert(tk.END, "\n--- Aggregation ---\n")
    output_text.insert(tk.END, "Aggregated fuzzy output calculated.\n")
    
    # Defuzzification
    O_crisp = np.sum(agg_heat * heating_range) / np.sum(agg_heat)
    output_text.insert(tk.END, "\n--- Defuzzification ---\n")
    output_text.insert(tk.END, f"Crisp Heating Output: {O_crisp:.2f}%\n")

# -----------------------------
# GUI layout
# -----------------------------
root = tk.Tk()
root.title("Fuzzy Logic Step-by-Step Demo")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="NSEW")

ttk.Label(frame, text="Temperature (0-40°C):").grid(row=0, column=0, sticky="W")
temp_entry = ttk.Entry(frame, width=10)
temp_entry.grid(row=0, column=1)
temp_entry.insert(0, "20")

ttk.Label(frame, text="Humidity (0-100%):").grid(row=1, column=0, sticky="W")
hum_entry = ttk.Entry(frame, width=10)
hum_entry.grid(row=1, column=1)
hum_entry.insert(0, "50")

compute_btn = ttk.Button(frame, text="Compute Fuzzy Steps", command=compute_fuzzy)
compute_btn.grid(row=2, column=0, columnspan=2, pady=10)

output_text = tk.Text(frame, width=70, height=25)
output_text.grid(row=3, column=0, columnspan=2)

root.mainloop()
