import matplotlib.pyplot as plt
import numpy as np



def plot_signals(x, y_true, y_pred, params):
    colors = ['b', 'g', 'r', 'm']
    linestyles = ['-', '--']

    print(params)

    # duration = params["durations"][0]

    # for i, (fb, fs, duration, resp, exc) in enumerate(zip(fbs, fss, durations, response, excitation)):
    fig, ax = plt.subplots( figsize=(20, 10))
    time = np.linspace(0, len(x)/params['fbs'], len(x))
    
    # Plot response and excitation on the same subplot
    ax.plot(time, y_pred, color=colors[0], linestyle=linestyles[1], alpha=0.4, label=f'y_pred')
    ax.plot(time, y_true, color=colors[1], linestyle=linestyles[1], alpha=0.4, label=f'y_true')
    ax.plot(time, x, color=colors[2], linestyle=linestyles[0],  label=f'Excitation')
    ax.set_title(f'Simulation vs Ground truth')
    ax.set_xlim([0, time[-1]])
    ax.legend()
    ax.grid(True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig.tight_layout()
    plt.savefig("plots/sim_v_truth.png", dpi=300)


if __name__ == "__main__":
    data = np.load("output/signals.npz",allow_pickle=True)

    samples = 100

    x = data["x"][0:samples]
    y_true = data["y_true"][0][0:samples]
    y_pred = data["y_pred"][0][0:samples]
    params = data["params"].item()

    # print(x.shape,y_true.shape,y_pred.shape)

    plot_signals(x, y_true, y_pred, params)


