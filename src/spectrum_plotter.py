# spectrum_plotter.py
import matplotlib.pyplot as plt
import os

def plot_spectrum(figure, A_list, B_list, filenames, backgrounds=None, xlabel="Velocity, mm/s", ylabel="Transmission, counts"):
    """
    Plot spectra on the given matplotlib figure.

    Parameters:
    - figure: matplotlib Figure object
    - A_list: list of x-data arrays
    - B_list: list of y-data arrays
    - filenames: list of filenames
    - backgrounds: list of background values for normalization (optional)
    - xlabel: label for x-axis (default: "Velocity, mm/s")
    - ylabel: label for y-axis (default: "Transmission, counts")
    """
    figure.clear()
    ax = figure.add_subplot(111)

    num_spectra = len(A_list)
    if num_spectra == 0:
        return

    # Determine colors
    if num_spectra == 1:
        colors = ['purple']
    else:
        from constants import numro
        ct = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato']
        ct.extend(ct)
        ct.extend(ct)
        ct.extend(ct)  # now it is 80
        model_colors = ct[:numro]  # One color per row, but repeat for spectra
        colors = (model_colors * ((num_spectra // len(model_colors)) + 1))[:num_spectra]

    # Determine X-axis range from the "biggest" spectrum (widest range)
    ranges = [max(A) - min(A) for A in A_list]
    biggest_idx = ranges.index(max(ranges))
    x_min, x_max = min(A_list[biggest_idx]), max(A_list[biggest_idx])

    # Plot each spectrum
    for i in range(num_spectra):
        A = A_list[i]
        B = B_list[i]
        color = colors[i]

        if backgrounds and len(backgrounds) > i and backgrounds[i] > 0:
            B_norm = B / backgrounds[i]
            ylabel = "Normalized transmission"
        else:
            B_norm = B

        ax.plot(A, B_norm, 'x', color=color, linestyle='None', markersize=5, label=os.path.basename(filenames[i]))

    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    if num_spectra == 1:
        ax.set_title(os.path.basename(filenames[0]), color='white')
        # Add right Y-axis with actual counts for single spectrum
        if backgrounds and len(backgrounds) > 0 and backgrounds[0] > 0:
            ax2 = ax.twinx()
            ax2.plot(A_list[0], B_list[0], 'x', color='purple', linestyle='None', markersize=5, alpha=0)
            ax2.set_ylabel('Transmission, counts', color='white')
            ax2.tick_params(axis='y', colors='white')
            ax2.set_facecolor('black')
            # Put ax on top so zooming affects ax
            ax2.set_zorder(ax.get_zorder() - 1)
            # Connect ylim change to update ax2 proportionally
            bg = backgrounds[0]
            def on_ylim_change(event):
                ylim = ax.get_ylim()
                ax2.set_ylim(ylim[0] * bg, ylim[1] * bg)
            ax.callbacks.connect('ylim_changed', on_ylim_change)
    else:
        ax.set_title(f"Multiple spectra ({num_spectra})", color='white')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.grid(True, color='white', linestyle=(0, (1, 10)), linewidth=0.5)  # Add grid
    ax.set_xlim(x_min, x_max)
    figure.tight_layout()  # Apply tight layout
    figure.canvas.draw()

    # Add legend if multiple spectra
    if num_spectra > 1:
        ax.legend(loc='best', fontsize='small', facecolor='black', edgecolor='white', labelcolor='white')