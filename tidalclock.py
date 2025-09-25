import requests
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

#Time format function
def format_time(dt, to_local=False):
    if to_local:
        dt = dt.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")
#Sine curve for predicting
def sine_func(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C


#Fetch last 24 hours of readings from Dover
now = datetime.now(timezone.utc)
start_time = now - timedelta(hours=24)
since_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
readings_url = (
    f"https://environment.data.gov.uk/flood-monitoring/id/measures/"
    f"E71624-level-tidal_level-Mean-15_min-m/readings?since={since_str}&_sorted&_limit=200"
)
response = requests.get(readings_url, timeout=10)
response.raise_for_status()
data = response.json().get("items", [])

if not data:
    raise Exception("No readings returned from the API for the last 24 hours.")

#Extract times and sea levels
times = np.array([datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")) for item in data])
levels = np.array([item["value"] for item in data])

high_indices, _ = find_peaks(levels, prominence=0.1, distance=20)  # 15-min intervals, ~5 hrs
low_indices, _ = find_peaks(-levels, prominence=0.1, distance=20)

# Get the last high/low
high_indices, _ = find_peaks(levels, distance=20, prominence=0.1)
low_indices, _  = find_peaks(-levels, distance=20, prominence=0.1)
last_high = max([(times[i], levels[i]) for i in high_indices if times[i] <= now], default=None)
last_low  = max([(times[i], levels[i]) for i in low_indices if times[i] <= now], default=None)

#Predict next high/low
t0 = times[0]
t_seconds = np.array([(t - t0).total_seconds() for t in times])
period_seconds = 12.4 * 3600  # 12.4 hours
omega_guess = 2 * np.pi / period_seconds
A_guess = (max(levels) - min(levels)) / 2
C_guess = np.mean(levels)
phi_guess = 0
popt, _ = curve_fit(sine_func, t_seconds, levels, p0=[A_guess, omega_guess, phi_guess, C_guess])
future_seconds = np.linspace(0, t_seconds[-1] + 24*3600, 5000)
future_levels = sine_func(future_seconds, *popt)
future_times = [t0 + timedelta(seconds=s) for s in future_seconds]

high_idx, _ = find_peaks(future_levels)
low_idx, _ = find_peaks(-future_levels)

next_high = None
next_low = None
for idx in high_idx:
    if future_times[idx] > times[-1]:
        next_high = (future_times[idx], future_levels[idx])
        break
for idx in low_idx:
    if future_times[idx] > times[-1]:
        next_low = (future_times[idx], future_levels[idx])
        break


if last_high and last_low:
    # Determine tide direction
    tide_direction = "coming in" if last_low[0] > last_high[0] else "going out"

else:
    # fallback to last two readings
    tide_direction = "coming in" if levels[-1] > levels[-2] else "going out"

# --- 5. Print results with nice formatting ---
print(f"Current tide: {tide_direction}")
if last_high:
    print("Last high tide:", format_time(last_high[0]), f"Level: {last_high[1]:.2f} m")
if last_low:
    print("Last low tide:", format_time(last_low[0]), f"Level: {last_low[1]:.2f} m")
if next_high:
    print("Predicted next high tide:", format_time(next_high[0]), f"Level: {next_high[1]:.2f} m")
if next_low:
    print("Predicted next low tide:", format_time(next_low[0]), f"Level: {next_low[1]:.2f} m")


# --- 6. Plot tide levels ---
plt.figure(figsize=(12,6))
plt.plot(times, levels, marker="o", markersize=3, label="Water level (m)")
if last_high:
    plt.plot(last_high[0], last_high[1], "ro", label="Last high")
if last_low:
    plt.plot(last_low[0], last_low[1], "bo", label="Last low")
plt.title("Tide levels at Dover (last 24 hours)")
plt.xlabel("Time")
plt.ylabel("Water level (m)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
