import matplotlib.pyplot as plt
import numpy as np



data = np.load("output/simulation_data.npz",allow_pickle=True)
response = data["response"]
excitation = data["excitation"]
params = data["params"].item()

fbs = params["fbs"]
fss = params["fss"]
durations = params["durations"]



# Create subplots per frequency (fb)
colors = ['b', 'g', 'r', 'm']
linestyles = ['-', '--']

for i, (fb, fs, duration, resp, exc) in enumerate(zip(fbs, fss, durations, response, excitation)):
    fig, ax = plt.subplots( figsize=(20, 5))
    time = np.linspace(0, duration, len(resp))
    
    # Plot response and excitation on the same subplot
    ax.plot(time, resp, color=colors[i], linestyle=linestyles[0], alpha=0.4, label=f'Response (fs={fs})')
    ax.plot(time, exc, color=colors[i], linestyle=linestyles[1],  label=f'Excitation (fs={fs})')
    ax.set_title(f'Response & Excitation (fb={fb})')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig.tight_layout()
    plt.savefig("plots/simulation.png", dpi=300)
