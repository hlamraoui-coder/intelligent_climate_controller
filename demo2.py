import numpy as np
import skfuzzy as fuzz

# -----------------------------
# Define fuzzy variables
# -----------------------------
temp_range = np.arange(0, 41, 1)   # 0°C to 40°C
humidity_range = np.arange(0, 101, 1)  # 0% to 100%
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

# Rules
rules = [
    {'temp': 'cold', 'hum': 'low', 'heat': 'high'},
    {'temp': 'cold', 'hum': 'comfortable', 'heat': 'low'},
    {'temp': 'comfortable', 'hum': 'low', 'heat': 'low'},
    {'temp': 'comfortable', 'hum': 'comfortable', 'heat': 'off'},
    {'temp': 'hot', 'hum': 'high', 'heat': 'off'},
    {'temp': 'hot', 'hum': 'comfortable', 'heat': 'low'},
    {'temp': 'hot', 'hum': 'low', 'heat': 'high'}
]

# Ask user for input
print("Temperature categories: cold, comfortable, hot")
current_temp_cat = input("Enter current temperature category: ").strip().lower()
desired_temp_cat = input("Enter desired temperature category: ").strip().lower()

# Fuzzification
print("\n--- Fuzzification ---")
def get_membership(temp_cat, temp_val):
    if temp_cat == 'cold':
        return fuzz.interp_membership(temp_range, temp_cold, temp_val)
    elif temp_cat == 'comfortable':
        return fuzz.interp_membership(temp_range, temp_comf, temp_val)
    elif temp_cat == 'hot':
        return fuzz.interp_membership(temp_range, temp_hot, temp_val)
    return 0

# For simplicity, assume numeric values for demonstration
current_temp_val = 10 if current_temp_cat=='cold' else 20 if current_temp_cat=='comfortable' else 30
desired_temp_val = 10 if desired_temp_cat=='cold' else 20 if desired_temp_cat=='comfortable' else 30

current_memberships = {
    'cold': get_membership('cold', current_temp_val),
    'comfortable': get_membership('comfortable', current_temp_val),
    'hot': get_membership('hot', current_temp_val)
}
desired_memberships = {
    'cold': get_membership('cold', desired_temp_val),
    'comfortable': get_membership('comfortable', desired_temp_val),
    'hot': get_membership('hot', desired_temp_val)
}

for k,v in current_memberships.items():
    print(f"Current temp '{k}': membership = {v:.2f}")
for k,v in desired_memberships.items():
    print(f"Desired temp '{k}': membership = {v:.2f}")

# Rule evaluation (firing strength)
print("\n--- Rule Activations ---")
def get_humidity_membership(hum_val):
    return {
        'low': fuzz.interp_membership(humidity_range, hum_low, hum_val),
        'comfortable': fuzz.interp_membership(humidity_range, hum_comf, hum_val),
        'high': fuzz.interp_membership(humidity_range, hum_high, hum_val)
    }

# Assume current humidity = 30% for example
hum_val = 30
hum_memberships = get_humidity_membership(hum_val)

fired_strengths = []
for i, rule in enumerate(rules):
    alpha_temp = current_memberships[rule['temp']]
    alpha_hum = hum_memberships[rule['hum']]
    firing_strength = np.min([alpha_temp, alpha_hum])
    fired_strengths.append((rule['heat'], firing_strength))
    print(f"Rule {i+1}: IF Temp is {rule['temp']} AND Humidity is {rule['hum']} THEN Heating is {rule['heat']}, firing strength = {firing_strength:.2f}")

# Aggregation
print("\n--- Aggregation ---")
# Initialize output membership arrays
agg_heat = np.zeros_like(heating_range, dtype=float)

for heat_type, alpha in fired_strengths:
    if heat_type == 'off':
        mf = np.fmin(alpha, heat_off)
    elif heat_type == 'low':
        mf = np.fmin(alpha, heat_low)
    elif heat_type == 'high':
        mf = np.fmin(alpha, heat_high)
    agg_heat = np.fmax(agg_heat, mf)  # max for aggregation

print("Aggregated fuzzy output prepared.")

# Defuzzification
O_crisp = np.sum(agg_heat * heating_range) / np.sum(agg_heat)
print("\n--- Defuzzification ---")
print(f"Crisp heating output = {O_crisp:.2f}%")
