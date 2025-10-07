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
            print(f"Plotting process: {process_name}")

            # process_data is an HDF5 group, we need to access the datasets

            # Get data arrays - process_data is HDF5 group, datasets are inside
            if process_name == 'central':
                loss = process_data.get('predicted_loss', None)
                grad_norm_sq = process_data.get('predicted_grad_norm_sq', None)
            else:
                loss = process_data.get('train_loss', None)
                grad_norm_sq = process_data.get('grad_norm_sq', None)
            hessian_eigs = process_data.get('hessian_eigs', None)

            # Determine number of steps based on available data
            step_counts = []
            for data in [loss, grad_norm_sq, hessian_eigs]:
                if data is not None:
                    step_counts.append(len(data))
            if step_counts:
                n_steps = max(step_counts)  # Use the largest dataset size
                print(f"  Found {n_steps} steps in data")
            else:
                print("  Warning: No data found for this process")
                continue

            steps = np.arange(n_steps)

            PROC_COLORS = {
                'discrete': 'tab:blue',
                'discrete_batch': 'tab:orange',
                'central': 'tab:green'
            }

            # Plot 1: Training loss (shared across all processes)
            if n_processes <= 1:
                target_loss_ax = axes[0]
            else:
                target_loss_ax = loss_ax

            if loss is not None:
                loss_vals = loss[:]
                mask = ~np.isnan(loss_vals)
                # 取当前进程的固定颜色
                c = PROC_COLORS.get(process_name, 'black')
                target_loss_ax.plot(steps[mask], loss_vals[mask],
                                    color=c, label=process_name, linewidth=2)
            else:
                print(f"Warning: No train_loss data for {process_name}")

            # Plot 2: Gradient norm (shared across all processes)
            if n_processes <= 1:
                target_grad_ax = axes[1]
            else:
                target_grad_ax = grad_ax

            if grad_norm_sq is not None:
                grad_vals = grad_norm_sq[:]
                mask = ~np.isnan(grad_vals)
                c = PROC_COLORS.get(process_name, 'black')
                target_grad_ax.plot(steps[mask], np.sqrt(grad_vals[mask]),
                                    color=c, label=process_name, linewidth=2)
            else:
                print(f"Warning: No gradient_sq_norm data for {process_name}")

            # Plot 3: Hessian eigenvalues (separate subplot per process)
            # 预先准备一套颜色，前 3 个按需要指定，后面的用默认颜色循环
            EIG_COLORS = ['blue', 'red', 'green']   # 0 号蓝，1 号红，2 号绿
            EIG_COLORS += plt.rcParams['axes.prop_cycle'].by_key()['color']  # 其余用默认色

            if n_processes <= 1:
                target_hess_ax = hessian_axes[0]
            else:
                target_hess_ax = hessian_axes[i]

            if (hessian_eigs is not None
                    and len(hessian_eigs.shape) >= 2
                    and hessian_eigs.shape[1] > 0):
                num_eigs = hessian_eigs.shape[1]
                for eig_idx in range(num_eigs):
                    eig_vals = hessian_eigs[:, eig_idx]
                    mask = ~np.isnan(eig_vals)
                    if np.sum(mask) > 0:
                        label = f'H e{eig_idx+1}'
                        # 关键：用颜色区分，不再用 linestyle 区分
                        color = EIG_COLORS[eig_idx % len(EIG_COLORS)]
                        target_hess_ax.plot(steps[mask], eig_vals[mask],
                                        color=color, label=label, linewidth=2)
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
