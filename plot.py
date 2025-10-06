#!/usr/bin/env python3
"""
Plotting script for central flows experiment results.

Usage:
python plot.py experiments/exp_id/data.hdf5 --processes discrete central discrete_batch
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_results(hdf5_path: str, processes: list[str]):
    """Plot experiment results from HDF5 file.

    Args:
        hdf5_path: Path to the data.hdf5 file
        processes: List of process names to plot (e.g., ['discrete', 'central'])
    """
    # Open HDF5 file
    with h5py.File(hdf5_path, "r", libver="latest", swmr=True) as datafile:
        print(f"Plotting results from: {hdf5_path}")
        print(f"Available processes: {list(datafile.keys())}")

        # Filter to only plot requested processes that exist
        available_processes = [p for p in processes if p in datafile]
        if not available_processes:
            print("None of the requested processes found in data file!")
            return

        # Check how many processes we have for layout
        n_processes = len(available_processes)

        # Set up the plot - Loss and Grad Norm in shared subplots, Hessian in separate subplots
        if n_processes <= 1:
            # Single process - use 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            hessian_axes = [axes[2]]
        else:
            # Multiple processes - Loss/Grad in 2 subplots, Hessian in N subplots
            fig, axes = plt.subplots(2 + n_processes, 1, figsize=(12, 4 + 3*n_processes))

            # First 2 subplots for Loss and Grad (shared)
            loss_ax = axes[0]
            grad_ax = axes[1]

            # Remaining subplots for Hessian (one per process)
            hessian_axes = axes[2:]

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        # Plot each process
        for i, process_name in enumerate(available_processes):
            color = colors[i % len(colors)]

            if process_name not in datafile:
                print(f"Warning: Process '{process_name}' not found in data")
                continue

            process_data = datafile[process_name]
            print(f"Plotting process: {process_name} ({len(process_data)} steps)")

            # Get data arrays
            steps = np.arange(len(process_data))
            loss = process_data.get('train_loss', None)
            grad_norm_sq = process_data.get('gradient_sq_norm', None)
            hessian_eigs = process_data.get('hessian_eigs', None)

            # Plot 1: Training loss (shared across all processes)
            if n_processes <= 1:
                target_loss_ax = axes[0]
            else:
                target_loss_ax = loss_ax

            if loss is not None:
                loss_vals = loss[:]
                mask = ~np.isnan(loss_vals)  # Filter out nan values
                target_loss_ax.plot(steps[mask], loss_vals[mask],
                                   color=color, label=f'{process_name} (loss)', linewidth=2)
            else:
                print(f"Warning: No train_loss data for {process_name}")

            # Plot 2: Gradient norm (shared across all processes)
            if n_processes <= 1:
                target_grad_ax = axes[1]
            else:
                target_grad_ax = grad_ax

            if grad_norm_sq is not None:
                grad_vals = grad_norm_sq[:]
                mask = ~np.isnan(grad_vals)  # Filter out nan values
                grad_norm = np.sqrt(grad_vals)  # Convert squared norm to norm
                target_grad_ax.plot(steps[mask], grad_norm[mask],
                                   color=color, label=f'{process_name} (grad norm)', linewidth=2)
            else:
                print(f"Warning: No gradient_sq_norm data for {process_name}")

            # Plot 3: Hessian eigenvalues (separate subplot per process)
            if n_processes <= 1:
                target_hess_ax = hessian_axes[0]
            else:
                target_hess_ax = hessian_axes[i]

            if hessian_eigs is not None and len(hessian_eigs.shape) >= 2 and hessian_eigs.shape[1] > 0:
                num_eigs = hessian_eigs.shape[1]
                for eig_idx in range(num_eigs):
                    eig_vals = hessian_eigs[:, eig_idx]
                    mask = ~np.isnan(eig_vals)  # Filter out nan values
                    if np.sum(mask) > 0:  # Only plot if we have valid data
                        label = f'H e{eig_idx+1}'
                        linestyle = '-' if eig_idx == 0 else '--' if eig_idx == 1 else '-.'
                        target_hess_ax.plot(steps[mask], eig_vals[mask],
                                           color='blue', linestyle=linestyle, label=label, linewidth=2)
                        print(f"  Plotted {np.sum(mask)} points for {process_name} eigenvalue {eig_idx+1}")
            else:
                print(f"Warning: No hessian_eigs data for {process_name}")

        # Configure shared loss plot
        if n_processes <= 1:
            axes[0].set_title('Training Loss vs Steps', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Training Loss', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        else:
            loss_ax.set_title('Training Loss vs Steps (All Processes)', fontsize=14, fontweight='bold')
            loss_ax.set_ylabel('Training Loss', fontsize=12)
            loss_ax.legend()
            loss_ax.grid(True, alpha=0.3)
            loss_ax.set_yscale('log')

        # Configure shared grad plot
        if n_processes <= 1:
            axes[1].set_title('Gradient Norm vs Steps', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Gradient Norm', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        else:
            grad_ax.set_title('Gradient Norm vs Steps (All Processes)', fontsize=14, fontweight='bold')
            grad_ax.set_ylabel('Gradient Norm', fontsize=12)
            grad_ax.legend()
            grad_ax.grid(True, alpha=0.3)
            grad_ax.set_yscale('log')

        # Configure Hessian plots (one per process)
        for i, (process_name, hess_ax) in enumerate(zip(available_processes, hessian_axes)):
            hess_ax.set_title(f'Hessian Eigenvalues - {process_name}', fontsize=14, fontweight='bold')
            hess_ax.set_ylabel('Eigenvalue', fontsize=12)
            hess_ax.legend()
            hess_ax.grid(True, alpha=0.3)

            # Set x-label only on the last Hessian subplot
            if i == len(hessian_axes) - 1:
                hess_ax.set_xlabel('Steps', fontsize=12)

        # Overall title and layout
        exp_id = Path(hdf5_path).parent.name
        fig.suptitle(f'Central Flows Experiment Results - {exp_id}\nProcesses: {", ".join(available_processes)}',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save plot
        output_path = Path(hdf5_path).parent / f'plots_{"_".join(available_processes)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

        # Show plot
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot central flows experiment results')
    parser.add_argument('hdf5_path', help='Path to the data.hdf5 file')
    parser.add_argument('--processes', nargs='+',
                       default=['discrete', 'central'],
                       help='List of processes to plot (default: discrete central)')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.hdf5_path).exists():
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        sys.exit(1)

    plot_results(args.hdf5_path, args.processes)


if __name__ == '__main__':
    main()
