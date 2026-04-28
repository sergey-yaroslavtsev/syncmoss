# spectrum_plotter.py
import matplotlib.pyplot as plt
import os
import numpy as np
import gc
from src.constants import numro, model_colors, number_of_baseline_parameters
from src.model_io import mod_len_def

# Default theme (original dark mode colors)
_DEFAULT_THEME = {
    'figure_facecolor': 'black',
    'axes_facecolor': 'black',
    'axes_text_color': 'white',
    'gridcolor': 'white',
    'legend_facecolor': 'black',
    'legend_edgecolor': 'white',
    'legend_textcolor': 'white',
}

_NON_SUBSPECTRUM_MODELS = {'Distr', 'Corr', 'Expression', 'Variables'}


def _tc(theme):
    """Resolve theme colors, returning dark defaults if theme is None."""
    return theme if theme is not None else _DEFAULT_THEME


def _subspectra_colors(model_colors, model=None):
    """Build flat color list for subspectra, skipping special models.
    
    model_colors[0] = baseline color, model_colors[i+1] = model[i] color.
    Returns one color per actual subspectrum (FS entry).
    """
    if model is None:
        return list(model_colors[1:]) if len(model_colors) > 1 else []
    colors = []
    for i, mod in enumerate(model):
        if mod not in _NON_SUBSPECTRUM_MODELS and mod != 'Nbaseline':
            idx = i + 1
            colors.append(model_colors[idx] if idx < len(model_colors) else 'white')
    return colors


def _subspectra_names(model):
    """Build flat list of model names for subspectra, skipping special models.
    
    Returns one name per actual subspectrum (FS entry), matching _subspectra_colors.
    """
    if model is None:
        return []
    names = []
    for mod in model:
        if mod not in _NON_SUBSPECTRUM_MODELS and mod != 'Nbaseline':
            names.append(mod)
    return names


def _style_axis(ax, tc):
    """Apply theme colors to a matplotlib axis."""
    ax.set_facecolor(tc['axes_facecolor'])
    ax.tick_params(colors=tc['axes_text_color'])
    for spine in ax.spines.values():
        spine.set_edgecolor(tc['axes_text_color'])


def calculate_baseline(p, x):
    """
    Calculate baseline from parameters and velocity array.
    
    Parameters:
    - p: parameter array (first 8 are baseline parameters)
    - x: velocity array
    
    Returns:
    - baseline array
    """
    return np.array(
        p[0] + p[3] * p[0]/100 * x + p[2] * p[0] / 10000 * (x - p[1])**2 + 
        p[6] * p[4] / 10000 * (x - p[5])**2 + p[4] + p[7] * p[4]/100 * x,
        dtype=float
    )


def calculate_z_order(FS):
    """
    Calculate z-order for subspectra based on their minimum values.
    Lower spectra get lower z-order (drawn first).
    
    Parameters:
    - FS: list of subspectra arrays
    
    Returns:
    - z_order array
    """
    if len(FS) == 0:
        return np.array([])
    
    v = np.array([len(FS)] * len(FS))
    for i in range(len(FS)):
        for k in range(len(FS)):
            if min(FS[i]) < min(FS[k]):
                v[i] -= 1
    return v


def plot_subspectra_with_positions(ax, x, y, FS, FS_pos, baseline, model_colors, z_order, color_offset=1, alpha=0.8, subspectra_names=None):
    """
    Plot subspectra with fill and position markers.
    
    Parameters:
    - ax: matplotlib axis
    - x: velocity array
    - y: experimental data (for position marker scaling)
    - FS: list of subspectra
    - FS_pos: list of position markers
    - baseline: baseline array
    - model_colors: color list
    - z_order: z-order array
    - color_offset: offset for color indexing (default: 1 to skip baseline)
    - alpha: transparency for fill (default: 0.8)
    - subspectra_names: optional list of names for legend labels
    
    Returns:
    - position_artists: list of position marker artists
    """
    position_artists = []
    skip_step = 0
    
    for i in range(len(FS)):
        # Get color
        color_idx = color_offset + i
        color = model_colors[color_idx] if color_idx < len(model_colors) else 'white'
        
        # Get z-order
        z = z_order[i] if i < len(z_order) else i
        
        # Get label for legend
        label = subspectra_names[i] if subspectra_names and i < len(subspectra_names) else None
        
        # Plot subspectrum with fill
        ax.plot(x, FS[i], color=color, zorder=z)
        ax.fill_between(x, baseline.astype(float), FS[i].astype(float), 
                        facecolor=color, alpha=alpha, zorder=z, label=label)
        
        # Plot position markers if available
        if FS_pos and len(FS_pos) > i and len(FS_pos[i][0]) > 0:
            minpos = min(FS_pos[i][0])
            maxpos = max(FS_pos[i][0])
            H_step = (max(y) - min(y)) * 0.04
            line_h = ax.plot([minpos, maxpos], 
                   [max(y) + H_step*(1+(i-skip_step)*2), max(y) + H_step*(1+(i-skip_step)*2)], 
                   color=color, zorder=z)
            position_artists.extend(line_h)
            for j in range(len(FS_pos[i][0])):
                line_v = ax.plot([FS_pos[i][0][j], FS_pos[i][0][j]], 
                       [max(y) + H_step*((i-skip_step)*2), max(y) + H_step*(1+(i-skip_step)*2)], 
                       color=color, zorder=z)
                position_artists.extend(line_v)
        else:
            skip_step += 1
    
    return position_artists


def plot_spectrum(figure, A_list, B_list, filenames, backgrounds=None, xlabel="Velocity, mm/s", ylabel="Transmission, counts", theme=None):
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
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    ax = figure.add_subplot(111)

    num_spectra = len(A_list)
    if num_spectra == 0:
        return

    # Determine colors
    if num_spectra == 1:
        colors = ['purple']
    else:
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

        # For multi-spectra: normalize if backgrounds available
        # For single spectrum: plot in counts
        if num_spectra > 1 and backgrounds and len(backgrounds) > i and backgrounds[i] > 0:
            # Multi-spectrum: plot normalized on main axis
            B_normalized = B / backgrounds[i]
            ax.plot(A, B_normalized, 'x', color=color, linestyle='None', markersize=5, label=os.path.basename(filenames[i]))
        else:
            # Single spectrum: plot in counts
            ax.plot(A, B, 'x', color=color, linestyle='None', markersize=5, label=os.path.basename(filenames[i]))

    ax.set_xlabel(xlabel, color=tc['axes_text_color'])
    
    if num_spectra == 1:
        # Single spectrum: counts on left, normalized on right
        ax.set_ylabel(ylabel, color=tc['axes_text_color'])
        ax.set_title(os.path.basename(filenames[0]), color=tc['axes_text_color'])
        # Add right Y-axis with normalized values for single spectrum
        if backgrounds and len(backgrounds) > 0 and backgrounds[0] > 0:
            bg = backgrounds[0]
            ax2 = ax.secondary_yaxis('right', functions=(lambda x: x / bg, lambda x: x * bg))
            ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
            ax2.tick_params(axis='y', colors=tc['axes_text_color'])
            for spine in ax2.spines.values():
                spine.set_edgecolor(tc['axes_text_color'])
    else:
        # Multi-spectrum: only normalized transmission (primary axis)
        ax.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
        ax.set_title(f"Multiple spectra ({num_spectra})", color=tc['axes_text_color'])
    _style_axis(ax, tc)
    ax.grid(True, color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    figure.tight_layout()
    figure.canvas.draw()

    # Add legend if multiple spectra
    if num_spectra > 1:
        legend = ax.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
        legend.set_visible(False)


def plot_calibration(figure, A, B, C, gridcolor='white', theme=None):
    """
    Plot calibration results on the given matplotlib figure.

    Parameters:
    - figure: matplotlib Figure object
    - A: x-data array (velocity)
    - B: y-data array (transmission counts)
    - C: fit data array
    - gridcolor: color for grid lines (default: 'white')
    - theme: theme dict for colors (default: None → dark mode)
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    ax = figure.add_subplot(111)
    ax.set_xlim(min(A), max(A))
    ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
    ax.plot(A, B, linestyle='None', marker='x', color='m', label='Data')
    ax.plot(A, C, color='r', label='Fit')
    ax.plot(A, B - C + min(B) - max(B - C), color='lime', label='Residual')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
    ax.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
    _style_axis(ax, tc)
    legend = ax.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
    legend.set_visible(False)
    figure.tight_layout()
    figure.canvas.draw()

def plot_model_with_nbaseline(figure, A, B, SPC_f, FS_all, FS_pos_all, p_all, model, model_colors, backgrounds=None, gridcolor='white', theme=None):
    """
    Plot model with Nbaseline separators - creates separate subplots for each spectrum.
    
    Parameters:
    - figure: matplotlib Figure object
    - A: concatenated x-data array
    - B: concatenated y-data array (experimental spectrum)
    - SPC_f: concatenated fitted spectrum data array
    - FS_all: list of lists of subspectra arrays for each spectrum section
    - FS_pos_all: list of lists of subspectra positions for each spectrum section
    - p_all: list of parameter arrays for each spectrum section
    - model: list of model names (includes Nbaseline separators)
    - model_colors: list of colors for each model row (including Nbaselines)
    - backgrounds: list of background values for normalization (optional)
    - gridcolor: color for grid lines (default: 'white')
    - theme: theme dict for colors (default: None → dark mode)
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    
    # Create filtered color list and names: remove colors at Nbaseline and special model positions
    # This builds a list where each entry corresponds to an actual subspectrum
    filtered_colors = [model_colors[0]]  # Keep experimental data color (index 0)
    filtered_names = []  # No name for baseline
    for i, mod in enumerate(model):
        if mod not in _NON_SUBSPECTRUM_MODELS and mod != 'Nbaseline':
            # i+1 because model_colors[0] is for experimental data
            if i + 1 < len(model_colors):
                filtered_colors.append(model_colors[i + 1])
            filtered_names.append(mod)
    
    # Split model into sections (without Nbaseline)
    model_sections = []
    start_idx = 0
    for i, m in enumerate(model):
        if m == 'Nbaseline':
            model_sections.append(model[start_idx:i])
            start_idx = i + 1
    model_sections.append(model[start_idx:])
    
    # Split concatenated arrays into separate spectra by detecting sign changes in step
    step_sign = np.sign(A[1] - A[0])
    x_separate = []
    y_separate = []
    start = 0
    for i in range(1, len(A)):
        if step_sign != np.sign(A[i] - A[i-1]):
            x_separate.append(A[start:i])
            y_separate.append(B[start:i])
            start = i
    x_separate.append(A[start:])
    y_separate.append(B[start:])
    
    # Split SPC_f the same way
    spc_separate = []
    start = 0
    for i in range(1, len(A)):
        if step_sign != np.sign(A[i] - A[i-1]):
            spc_separate.append(SPC_f[start:i])
            start = i
    spc_separate.append(SPC_f[start:])
    
    num_spectra = len(x_separate)
    
    # Collect position artists from all spectra
    position_artists_all = []
    
    # Create subplots - one for each spectrum
    for spc_idx in range(num_spectra):
        ax = figure.add_subplot(1, num_spectra, spc_idx + 1)
        
        x = x_separate[spc_idx]
        y = y_separate[spc_idx]
        spc = spc_separate[spc_idx]
        p = p_all[spc_idx] if spc_idx < len(p_all) else p_all[0]
        FS = FS_all[spc_idx] if spc_idx < len(FS_all) else []
        FS_pos = FS_pos_all[spc_idx] if spc_idx < len(FS_pos_all) else []
        
        # Determine if we should normalize this spectrum
        normalize = backgrounds and spc_idx < len(backgrounds) and backgrounds[spc_idx] > 0
        bg = backgrounds[spc_idx] if normalize else 1.0
        
        # Always plot in counts (no normalization applied to plotted data)
        y_plot = y
        spc_plot = spc
        
        # Set axis limits and grid
        ax.set_xlim(min(x), max(x))
        ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
        
        # Calculate z-order for subspectra
        if len(FS) > 0:
            v = np.array([len(FS)] * len(FS))
            for i in range(len(FS)):
                for k in range(len(FS)):
                    if min(FS[i]) < min(FS[k]):
                        v[i] -= 1
            
            # Plot each subspectrum with fill
            baseline = p[0] + p[3] * p[0]/100 * x + p[2] * p[0] / 10000 * (x - p[1])**2 + \
                       p[6] * p[4] / 10000 * (x - p[5])**2 + p[4] + p[7] * p[4]/100 * x
            baseline_plot = baseline  # Always use counts
            
            distri_counter = 0
            skip_step = 0
            position_artists = []  # Collect position artists for this spectrum
            
            # Count how many subspectra come before this section
            subspectra_before = 0
            for prev_idx in range(spc_idx):
                subspectra_before += len(FS_all[prev_idx])
            
            for i in range(len(FS)):
                FS_i_plot = FS[i]  # Always use counts
                # Each subspectrum gets its own color from filtered_colors
                # filtered_colors has Nbaseline and special model positions removed
                color_idx = 1 + subspectra_before + i
                color = filtered_colors[color_idx] if color_idx < len(filtered_colors) else 'white'
                name_idx = subspectra_before + i
                label = filtered_names[name_idx] if name_idx < len(filtered_names) else None
                ax.plot(x, FS_i_plot, color=color, zorder=v[i])
                ax.fill_between(x, baseline_plot.astype(float), FS_i_plot.astype(float), 
                              facecolor=color, alpha=0.8, zorder=v[i], label=label)
                
                # Plot position markers if available
                if len(FS_pos) > i and len(FS_pos[i][0]) > 0:
                    minpos = min(FS_pos[i][0])
                    maxpos = max(FS_pos[i][0])
                    H_step = (max(y_plot) - min(y_plot)) * 0.04
                    line_h = ax.plot([minpos, maxpos], 
                           [max(y_plot) + H_step*(1+(i-skip_step)*2), max(y_plot) + H_step*(1+(i-skip_step)*2)], 
                           color=color, zorder=v[i])
                    position_artists.extend(line_h)
                    for j in range(len(FS_pos[i][0])):
                        line_v = ax.plot([FS_pos[i][0][j], FS_pos[i][0][j]], 
                               [max(y_plot) + H_step*((i-skip_step)*2), max(y_plot) + H_step*(1+(i-skip_step)*2)], 
                               color=color, zorder=v[i])
                        position_artists.extend(line_v)
                else:
                    skip_step += 1
            
            # Add this spectrum's position artists to the collection
            position_artists_all.append(position_artists)
        
        # Plot residual
        residual = y_plot - spc_plot + min(y_plot) - max(y_plot - spc_plot)
        ax.plot(x, residual, color='lime', label='Residual')
        ax.plot(x, y_plot - y_plot + min(y_plot) - max(y_plot - spc_plot), linestyle='--', color=tc['gridcolor'])
        
        # Plot fit and data
        ax.plot(x, spc_plot, color='r', zorder=len(FS)+2 if FS else 2, label='Fit')
        ax.plot(x, y_plot, linestyle='None', marker='x', color='m', zorder=len(FS)+1 if FS else 1, label='Data')
        
        # Formatting
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
        if spc_idx == 0:
            ax.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
        _style_axis(ax, tc)
        legend = ax.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
        legend.set_visible(False)
        
        # Add right Y-axis with normalized values if this spectrum has background
        if normalize:
            ax2 = ax.secondary_yaxis('right', functions=(lambda x, bg=bg: x / bg, lambda x, bg=bg: x * bg))
            if spc_idx == num_spectra - 1:  # Only label on last subplot
                ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
            ax2.tick_params(axis='y', colors=tc['axes_text_color'])
            for spine in ax2.spines.values():
                spine.set_edgecolor(tc['axes_text_color'])
    
    figure.tight_layout()
    figure.canvas.draw()
    
    # Collect all position artists from all sections
    all_position_artists = []
    for artists in position_artists_all:
        all_position_artists.extend(artists)
    return all_position_artists


def plot_simultaneous_fitting_result(figure, A_list, B_list, SPC_f_list, FS_list, FS_pos_list, p_all, begining_spc, model_colors, chi2, spectrum_files, dir_path, z_order=None, gridcolor='white', theme=None, model=None):
    """
    Plot simultaneous fitting results with multiple spectra in separate subplots.
    
    Parameters:
    - figure: matplotlib Figure object
    - A_list, B_list, SPC_f_list, FS_list, FS_pos_list: data per spectrum
    - p_all: full parameter array
    - begining_spc: parameter start indices per spectrum
    - model_colors: colors for each model row
    - chi2: chi-squared value
    - spectrum_files: file paths
    - dir_path: output directory
    - z_order: optional custom z-order (flattened)
    - gridcolor: grid line color
    - theme: theme dict for colors
    - model: model list with Nbaseline separators for correct color mapping
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    
    num_spectra = len(A_list)
    
    # Collect position artists from all spectra
    position_artists_list = []
    
    # Build per-section color and name lists (skipping Distr/Corr/Expression/Variables)
    if model is not None:
        section_colors = []
        section_names = []
        current_colors = []
        current_names = []
        ci = 1  # skip baseline at model_colors[0]
        for mod in model:
            if mod == 'Nbaseline':
                section_colors.append(current_colors)
                section_names.append(current_names)
                current_colors = []
                current_names = []
                ci += 1  # skip Nbaseline color
            else:
                if mod not in _NON_SUBSPECTRUM_MODELS:
                    current_colors.append(model_colors[ci] if ci < len(model_colors) else 'white')
                    current_names.append(mod)
                ci += 1
        section_colors.append(current_colors)
        section_names.append(current_names)
    else:
        section_colors = None
        section_names = None
    
    z_offset = 0  # Track position in flattened z_order array
    
    # Create subplots - one for each spectrum
    for spc_idx in range(num_spectra):
        ax = figure.add_subplot(1, num_spectra, spc_idx + 1)
        
        x = A_list[spc_idx]
        y = B_list[spc_idx]
        spc = SPC_f_list[spc_idx]
        FS = FS_list[spc_idx] if spc_idx < len(FS_list) else []
        FS_pos = FS_pos_list[spc_idx] if spc_idx < len(FS_pos_list) else []
        
        # Get parameters for this spectrum
        if spc_idx < len(begining_spc) - 1:
            p = p_all[begining_spc[spc_idx]:begining_spc[spc_idx + 1]]
        else:
            p = p_all[begining_spc[spc_idx]:]
        
        # Set axis limits and grid
        ax.set_xlim(min(x), max(x))
        ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
        
        # Calculate or use provided z-order for subspectra
        position_artists = []
        if len(FS) > 0:
            baseline = calculate_baseline(p, x)
            
            # Calculate z-order if not provided
            if z_order is None:
                v = calculate_z_order(FS)
            else:
                # Use custom z-order from flattened array (ensure all are integers)
                v = [int(z_order[z_offset + i]) if z_offset + i < len(z_order) else i for i in range(len(FS))]
            
            skip_step = 0
            
            for i in range(len(FS)):
                # Get color for this subspectrum
                if section_colors is not None and spc_idx < len(section_colors):
                    sc = section_colors[spc_idx]
                    color = sc[i] if i < len(sc) else 'white'
                else:
                    # Fallback: direct indexing (old behavior)
                    color = model_colors[1 + i] if (1 + i) < len(model_colors) else 'white'
                
                # Get label for legend
                label = None
                if section_names is not None and spc_idx < len(section_names):
                    sn = section_names[spc_idx]
                    label = sn[i] if i < len(sn) else None
                
                ax.plot(x, FS[i], color=color, zorder=v[i])
                ax.fill_between(x, baseline.astype(float), FS[i].astype(float), 
                              facecolor=color, alpha=0.8, zorder=v[i], label=label)
                
                # Plot position markers if available
                if len(FS_pos) > i and len(FS_pos[i][0]) > 0:
                    minpos = min(FS_pos[i][0])
                    maxpos = max(FS_pos[i][0])
                    H_step = (max(y) - min(y)) * 0.04
                    line_h = ax.plot([minpos, maxpos], 
                           [max(y) + H_step*(1+(i-skip_step)*2), max(y) + H_step*(1+(i-skip_step)*2)], 
                           color=color, zorder=v[i])
                    position_artists.extend(line_h)
                    for j in range(len(FS_pos[i][0])):
                        line_v = ax.plot([FS_pos[i][0][j], FS_pos[i][0][j]], 
                               [max(y) + H_step*((i-skip_step)*2), max(y) + H_step*(1+(i-skip_step)*2)], 
                               color=color, zorder=v[i])
                        position_artists.extend(line_v)
                else:
                    skip_step += 1
            
            position_artists_list.append(position_artists)
        else:
            v = []
            position_artists_list.append([])
        
        # Plot residual
        residual = y - spc + min(y) - max(y - spc)
        ax.plot(x, residual, color='lime', label='Residual')
        ax.plot(x, y - y + min(y) - max(y - spc), linestyle='--', color=tc['gridcolor'])
        
        # Plot fit and data with higher z-order
        max_z = int(max(v)) if len(v) > 0 else len(FS)
        ax.plot(x, spc, color='r', zorder=max_z+2, label='Fit')
        ax.plot(x, y, linestyle='None', marker='x', color='m', zorder=max_z+1, label='Data')
        
        # Add spectrum filename at bottom
        if spc_idx < len(spectrum_files):
            ax.text(0, -0.1, os.path.basename(spectrum_files[spc_idx]), 
                   horizontalalignment='left', verticalalignment='center', 
                   color='m', transform=ax.transAxes)
        
        # Add chi2 title to middle subplot
        if spc_idx == int(num_spectra / 2):
            ax.set_title(f'χ² = {chi2:.3f}', y=1, color='r')
        
        # Formatting
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
        if spc_idx == 0:
            ax.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
        _style_axis(ax, tc)
        legend = ax.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
        legend.set_visible(False)
        
        # Add right Y-axis with normalized values
        if p[2] == 0 and p[6] == 0:  # Only if no quadratic baseline
            bg = p[0] + p[4]
            ax2 = ax.secondary_yaxis('right', functions=(lambda x, bg=bg: x / bg, lambda x, bg=bg: x * bg))
            if spc_idx == num_spectra - 1:  # Only label on last subplot
                ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
            ax2.tick_params(axis='y', colors=tc['axes_text_color'])
            for spine in ax2.spines.values():
                spine.set_edgecolor(tc['axes_text_color'])
        
        # Update z-order offset for next spectrum
        z_offset += len(FS)
    
    figure.tight_layout()
    figure.canvas.draw()
    
    # Save images only if not in replot mode (z_order was None initially)
    svg_path = None
    png_path = None
    if z_order is None:  # Original plot, not replot
        svg_path = os.path.join(dir_path, 'result.svg')
        png_path = os.path.join(dir_path, 'result.png')
        figure.savefig(svg_path, bbox_inches='tight', facecolor=tc['figure_facecolor'])
        figure.savefig(png_path, bbox_inches='tight', facecolor=tc['figure_facecolor'], dpi=300)
    
    return svg_path, png_path, position_artists_list


def plot_model(figure, A, B, SPC_f, FS, FS_pos, p, model_colors, backgrounds=None, gridcolor='white', theme=None, model=None):
    """
    Plot model with subspectra on the given matplotlib figure.
    
    Parameters:
    - figure: matplotlib Figure object
    - A: x-data array (velocity)
    - B: y-data array (experimental spectrum)
    - SPC_f: fitted spectrum data array
    - FS: list of subspectra arrays
    - FS_pos: list of subspectra positions (for line markers)
    - p: parameter array (first 8 are baseline parameters)
    - model_colors: list of colors for each model row
    - backgrounds: list of background values for normalization (optional)
    - gridcolor: color for grid lines (default: 'white')
    - theme: theme dict for colors (default: None → dark mode)
    - model: model list for correct color mapping (skips Distr/Corr etc.)
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    ax = figure.add_subplot(111)
    
    # Build subspectra color list and names (skipping special models)
    sub_colors = _subspectra_colors(model_colors, model)
    sub_names = _subspectra_names(model)
    
    # Determine if we should normalize (single spectrum with background)
    normalize = backgrounds and len(backgrounds) > 0 and backgrounds[0] > 0
    bg = backgrounds[0] if normalize else 1.0
    
    # Always plot in counts on main axis (no normalization applied to plotted data)
    B_plot = B
    SPC_f_plot = SPC_f
    
    # Set axis limits and grid
    ax.set_xlim(min(A), max(A))
    ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
    
    # Calculate z-order for subspectra (lower spectra should be drawn first)
    if len(FS) > 0:
        v = np.array([len(FS)] * len(FS))
        for i in range(len(FS)):
            for k in range(len(FS)):
                if min(FS[i]) < min(FS[k]):
                    v[i] -= 1
        
        # Plot each subspectrum with fill
        baseline = p[0] + p[3] * p[0]/100 * A + p[2] * p[0] / 10000 * (A - p[1])**2 + \
                   p[6] * p[4] / 10000 * (A - p[5])**2 + p[4] + p[7] * p[4]/100 * A
        baseline_plot = baseline
        
        skip_step = 0
        position_artists = []
        for i in range(len(FS)):
            FS_i_plot = FS[i]
            color = sub_colors[i] if i < len(sub_colors) else 'white'
            label = sub_names[i] if i < len(sub_names) else None
            ax.plot(A, FS_i_plot, color=color, zorder=v[i])
            ax.fill_between(A, baseline_plot.astype(float), FS_i_plot.astype(float), 
                          facecolor=color, alpha=0.8, zorder=v[i], label=label)
            
            # Plot position markers if available
            if len(FS_pos) > i and len(FS_pos[i][0]) > 0:
                minpos = min(FS_pos[i][0])
                maxpos = max(FS_pos[i][0])
                H_step = (max(B_plot) - min(B_plot)) * 0.04
                line_h = ax.plot([minpos, maxpos], 
                       [max(B_plot) + H_step*(1+(i-skip_step)*2), max(B_plot) + H_step*(1+(i-skip_step)*2)], 
                       color=color, zorder=v[i])
                position_artists.extend(line_h)
                for j in range(len(FS_pos[i][0])):
                    line_v = ax.plot([FS_pos[i][0][j], FS_pos[i][0][j]], 
                           [max(B_plot) + H_step*((i-skip_step)*2), max(B_plot) + H_step*(1+(i-skip_step)*2)], 
                           color=color, zorder=v[i])
                    position_artists.extend(line_v)
            else:
                skip_step += 1
    
    # Plot residual
    residual = B_plot - SPC_f_plot + min(B_plot) - max(B_plot - SPC_f_plot)
    ax.plot(A, residual, color='lime', label='Residual')
    ax.plot(A, B_plot - B_plot + min(B_plot) - max(B_plot - SPC_f_plot), linestyle='--', color=tc['gridcolor'])
    
    # Plot fit and data
    ax.plot(A, SPC_f_plot, color='r', zorder=len(FS)+2, label='Fit')
    ax.plot(A, B_plot, linestyle='None', marker='x', color='m', zorder=len(FS)+1, label='Data')
    
    # Formatting
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
    ax.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
    
    # Add secondary Y-axis with normalized values if we have background
    if normalize:
        ax2 = ax.secondary_yaxis('right', functions=(lambda x: x / bg, lambda x: x * bg))
        ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
        ax2.tick_params(axis='y', colors=tc['axes_text_color'])
        for spine in ax2.spines.values():
            spine.set_edgecolor(tc['axes_text_color'])
    
    _style_axis(ax, tc)
    legend = ax.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
    legend.set_visible(False)
    
    figure.tight_layout()
    figure.canvas.draw()
    
    return position_artists if 'position_artists' in locals() else []


def plot_instrumental_result(figure, A, B, F, F2, p, hi2, filepath, dir_path, gridcolor='gray', theme=None):
    """
    Plot instrumental function fitting results on the given figure and save to files.
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    ax1 = figure.add_subplot(111)
    ax1.set_xlim(min(A), max(A))
    ax1.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
    
    # Plot fitted spectrum
    ax1.plot(A, F, color='r')
    
    # Fill between fitted spectrum and baseline
    baseline = np.array(p[0] + p[3] * p[0]/10**2 * A + p[2] * p[0] / 10**4 * (A - p[1])**2 + 
                       p[6] * p[4] / 10**4 * (A - p[5]) ** 2 + p[4] + p[7] * p[4]/10**2 * A, dtype=float)
    ax1.fill_between(A, F.astype(float), baseline, color='r', alpha=1, zorder=2)
    
    # Plot experimental data
    ax1.plot(A, B, linestyle='None', marker='x', color='m')
    
    # Plot residuals
    residual_offset = min(B) - max(B - F)
    ax1.plot(A, B - F + residual_offset, color='lime')
    ax1.plot(A, B - B + residual_offset, linestyle='--', color=tc['gridcolor'])
    
    # Plot high-resolution difference
    hires_offset = residual_offset + min(B - F) - max(F2 - F)
    ax1.plot(A, F2 - F + hires_offset, color='cyan')
    
    # Add file path annotation
    ax1.text(0, -0.1, os.path.abspath(filepath), horizontalalignment='left', 
            verticalalignment='center', color='m', transform=ax1.transAxes)
    
    # Add chi-squared title
    ax1.set_title('χ² = %.3f' % hi2, y=1, color='r')
    
    # Style axis
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
    ax1.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
    _style_axis(ax1, tc)
    
    # Add secondary y-axis for normalized scale
    bg = p[0]
    if bg > 0:
        ax2 = ax1.secondary_yaxis('right', functions=(lambda x: x / bg, lambda x: x * bg))
        ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
        ax2.tick_params(axis='y', colors=tc['axes_text_color'])
        for spine in ax2.spines.values():
            spine.set_edgecolor(tc['axes_text_color'])
    
    # Save plots
    result_svg = os.path.join(dir_path, 'result.svg')
    figure.savefig(result_svg, bbox_inches='tight', facecolor=tc['figure_facecolor'], dpi=300)
    
    result_png = os.path.join(dir_path, 'result.png')
    figure.savefig(result_png, bbox_inches='tight', facecolor=tc['figure_facecolor'], dpi=300)
    
    print(f"[DEBUG] Saved plots: {result_svg}, {result_png}")
    
    # Update layout
    figure.tight_layout()
    
    return result_svg, result_png


def plot_fitting_result(figure, A, B, SPC_f, FS, FS_pos, p, model_colors, hi2, filepath, dir_path, z_order=None, gridcolor='gray', theme=None, model=None):
    """
    Plot spectrum fitting results on the given figure and save to files.
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    ax1 = figure.add_subplot(111)
    ax1.set_xlim(min(A), max(A))
    ax1.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=1)
    
    # Calculate baseline and z-order
    baseline = calculate_baseline(p, A)
    z_order_was_none = z_order is None
    if z_order is None:
        z_order = calculate_z_order(FS)
    
    # Build subspectra colors and names (skipping special models)
    sub_colors = _subspectra_colors(model_colors, model)
    sub_names = _subspectra_names(model)
    position_artists = plot_subspectra_with_positions(
        ax1, A, B, FS, FS_pos, baseline, sub_colors, z_order, 
        color_offset=0, alpha=0.8, subspectra_names=sub_names
    )
    
    # Plot fit and data with higher z-order
    max_z = int(max(z_order)) if len(z_order) > 0 else len(FS)
    ax1.plot(A, SPC_f, color='r', zorder=max_z+2, label='Fit')
    ax1.plot(A, B, linestyle='None', marker='x', color='m', zorder=max_z+1, label='Data')
    
    # Plot residuals
    residual_offset = min(B) - max(B - SPC_f)
    ax1.plot(A, B - SPC_f + residual_offset, color='lime', label='Residual')
    ax1.plot(A, B - B + residual_offset, linestyle='--', color=tc['gridcolor'])
    
    # Add labels
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_ylabel('Transmission, counts', color=tc['axes_text_color'])
    ax1.set_xlabel('Velocity, mm/s', color=tc['axes_text_color'])
    _style_axis(ax1, tc)
    legend = ax1.legend(loc='lower left', fontsize='small', facecolor=tc['legend_facecolor'], edgecolor=tc['legend_edgecolor'], labelcolor=tc['legend_textcolor'])
    legend.set_visible(False)
    
    # Add file path annotation
    ax1.text(0, -0.1, os.path.basename(filepath), horizontalalignment='left', 
            verticalalignment='center', color='m', transform=ax1.transAxes)
    
    # Add chi-squared title
    ax1.set_title('χ² = %.3f' % hi2, y=1, color='r')
    
    # Add secondary y-axis for normalized scale if baseline is non-zero
    if p[2] == 0 and p[6] == 0:
        bg = p[0] + p[4]
        if bg > 0:
            ax2 = ax1.secondary_yaxis('right', functions=(lambda x: x / bg, lambda x: x * bg))
            ax2.set_ylabel('Normalized transmission', color=tc['axes_text_color'])
            ax2.tick_params(axis='y', colors=tc['axes_text_color'])
            for spine in ax2.spines.values():
                spine.set_edgecolor(tc['axes_text_color'])
    
    # Save plots only if not in replot mode (z_order was None initially)
    result_svg = None
    result_png = None
    if z_order_was_none:
        result_svg = os.path.join(dir_path, 'result.svg')
        figure.savefig(result_svg, bbox_inches='tight', facecolor=tc['figure_facecolor'], dpi=300)
        
        result_png = os.path.join(dir_path, 'result.png')
        figure.savefig(result_png, bbox_inches='tight', facecolor=tc['figure_facecolor'], dpi=300)
        
        print(f"[DEBUG] Saved fitting plots: {result_svg}, {result_png}")
    
    # Update layout
    figure.tight_layout()
    
    return result_svg, result_png, position_artists


def plot_distribution(figure, model, p, Distri, Cor, parameter_names, model_colors=None, gridcolor='gray', theme=None):
    """
    Plot distribution probability densities and correlations on the figure.
    
    Parameters:
    - figure: matplotlib Figure object
    - model: list of model names (full model including Distr/Corr)
    - p: fitted parameter array
    - Distri: list of distribution expressions (already substituted)
    - Cor: list of correlation expressions (already substituted)
    - parameter_names: list of lists of parameter names per component
    - model_colors: list of colors per model row (for dot markers)
    - gridcolor: color for grid lines
    - theme: theme dict for colors
    
    Returns:
    - True if distribution was plotted, False if no distribution found
    """
    tc = _tc(theme)
    figure.clear()
    figure.patch.set_facecolor(tc['figure_facecolor'])
    
    model_np = np.array(model)
    distr_indices = np.where(model_np == 'Distr')[0]
    
    if len(distr_indices) == 0:
        return False
    
    # Append sentinel for range checking
    Distri_D = np.append(distr_indices, len(model_np) * 2)
    Corr_D = np.where(model_np == 'Corr')[0]
    
    # Count correlations per distribution
    corr_counts = []
    for j in range(len(Distri_D) - 1):
        cnt = 0
        for k in range(len(Corr_D)):
            if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j + 1]:
                cnt += 1
        corr_counts.append(cnt)
    
    num_distr = len(distr_indices)
    
    # Helper: get model color for a model index
    def _get_model_color(model_idx):
        if model_colors is not None and (model_idx + 1) < len(model_colors):
            return model_colors[model_idx + 1]  # +1 because [0] is baseline
        return tc['axes_text_color']
    
    Di_idx = 0  # index into Distri list
    Co_idx = 0  # index into Cor list
    
    for j in range(num_distr):
        n_corr = corr_counts[j]
        # Total columns for this row: 1 (distribution) + n_corr (correlations)
        # Use GridSpec for flexible subplot sizing
        total_cols = 1 + n_corr if n_corr > 0 else 1
        
        # Calculate parameter offset for this distribution
        Vnum = number_of_baseline_parameters
        for k in range(int(Distri_D[j])):
            Vnum += mod_len_def(model[k], include_special=True)
        
        # Dot color follows the Distr model row color
        dot_color = _get_model_color(int(Distri_D[j]))
        
        # Distribution subplot - spans full width if no correlations
        if total_cols == 1:
            ax = figure.add_subplot(num_distr, 1, j + 1)
        else:
            ax = figure.add_subplot(num_distr, total_cols, j * total_cols + 1)
        
        # X range from distribution parameters
        n_points = int(p[Vnum + 3])
        X_discrete = np.linspace(p[Vnum + 1], p[Vnum + 2], n_points)
        X_smooth = np.linspace(p[Vnum + 1], p[Vnum + 2], 1024)
        
        # Evaluate distribution expression
        try:
            X = X_discrete
            Y_discrete = eval(Distri[Di_idx]) + 0 * X
            X = X_smooth
            Y_smooth = eval(Distri[Di_idx]) + 0 * X
        except Exception as e:
            print(f"Error evaluating distribution expression: {e}")
            Di_idx += 1
            continue
        
        # Normalize
        S = Y_smooth.sum()
        if S == 0:
            S = 1
        Y_smooth_norm = Y_smooth / S
        Y_discrete_norm = Y_discrete / S
        
        # Plot: smooth curve in BLUE, discrete dots in parent model color
        ax.plot(X_smooth, Y_smooth_norm, color='blue', linewidth=1.5)
        ax.plot(X_discrete, Y_discrete_norm, marker='h', linestyle='', color=dot_color, markersize=6)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=0.5)
        _style_axis(ax, tc)
        
        # Y-axis label
        if j == num_distr // 2:
            ax.set_ylabel('Probability Density', color=tc['axes_text_color'])
        
        # X-axis label
        xlabel = _get_distribution_xlabel(model, Distri_D[j], parameter_names)
        ax.set_xlabel(xlabel, color=tc['axes_text_color'])
        
        # Plot correlations for this distribution
        Co_n = 0
        for k in range(len(Corr_D)):
            if Corr_D[k] > Distri_D[j] and Corr_D[k] < Distri_D[j + 1]:
                ax_corr = figure.add_subplot(num_distr, total_cols, j * total_cols + 2 + Co_n)
                
                # Correlation dot color follows the Corr model row color
                corr_dot_color = _get_model_color(int(Corr_D[k]))
                
                try:
                    X = X_smooth
                    YY_smooth = eval(Cor[Co_idx + Co_n]) + 0 * X
                    ax_corr.plot(YY_smooth, Y_smooth_norm, color='red', linewidth=1.5)
                    
                    X = X_discrete
                    YY_discrete = eval(Cor[Co_idx + Co_n]) + 0 * X
                    ax_corr.plot(YY_discrete, Y_discrete_norm, marker='H', linestyle='', color=corr_dot_color, markersize=6)
                except Exception as e:
                    print(f"Error evaluating correlation expression: {e}")
                
                ax_corr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax_corr.grid(color=tc['gridcolor'], linestyle=(0, (1, 10)), linewidth=0.5)
                _style_axis(ax_corr, tc)
                
                corr_xlabel = _get_correlation_xlabel(model, Distri_D[j], Co_n + 1, parameter_names)
                ax_corr.set_xlabel(corr_xlabel, color=tc['axes_text_color'])
                
                Co_n += 1
        
        Di_idx += 1
        Co_idx += Co_n
    
    figure.tight_layout()
    figure.canvas.draw()
    return True


def _get_distribution_xlabel(model, distr_model_idx, parameter_names):
    """Get x-axis label for a distribution plot."""
    parent_idx = int(distr_model_idx)
    for k in range(1, len(model)):
        candidate = parent_idx - k
        if candidate < 0:
            break
        if model[candidate] not in _NON_SUBSPECTRUM_MODELS:
            parent_idx = candidate
            break
    try:
        if parent_idx + 1 < len(parameter_names):
            return f"{model[parent_idx]} parameter"
    except (IndexError, ValueError):
        pass
    return "Parameter"


def _get_correlation_xlabel(model, distr_model_idx, corr_offset, parameter_names):
    """Get x-axis label for a correlation plot."""
    parent_idx = int(distr_model_idx)
    for k in range(1, len(model)):
        candidate = parent_idx - k
        if candidate < 0:
            break
        if model[candidate] not in _NON_SUBSPECTRUM_MODELS:
            parent_idx = candidate
            break
    try:
        if parent_idx + 1 < len(parameter_names):
            return f"{model[parent_idx]} corr. parameter"
    except (IndexError, ValueError):
        pass
    return "Correlated parameter"


