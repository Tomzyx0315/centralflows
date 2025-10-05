#!/usr/bin/env python3
"""
可视化 Central Flows 训练曲线的脚本

从保存的 HDF5 文件中读取数据，绘制 full-batch, mini-batch 和 central flow 的对比曲线
"""

import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_data(experiment_path):
    """从实验文件夹加载所有数据"""
    h5_file = Path(experiment_path) / "data.hdf5"

    if not h5_file.exists():
        raise FileNotFoundError(f"Data file not found: {h5_file}")

    data = {}

    with h5py.File(h5_file, "r", libver="latest", swmr=True) as f:
        # 打印所有可用的key来调试
        print(f"Found HDF5 keys: {list(f.keys())}")

        # 查找所有根级别的键
        for key in f.keys():
            if key.startswith("discrete_full") or key == "discrete_full":
                data["full_batch"] = extract_process_data(f, key)
            elif key.startswith("discrete_mini") or key == "discrete_mini":
                data["mini_batch"] = extract_process_data(f, key)
            elif key.startswith("central") or key == "central":
                data["central_flow"] = extract_process_data(f, key)

        # 如果没有找到带后缀的数据，尝试不带后缀的
        if "full_batch" not in data and "discrete" in f:
            data["full_batch"] = extract_process_data(f, "discrete")

    return data


def extract_process_data(f, process_key):
    """从HDF5文件中提取一个进程的数据"""
    if process_key not in f:
        return None

    process_group = f[process_key]
    data = {}

    # 收集所有step的数据
    steps = []
    losses = []
    grad_norms = []
    hessian_eigs = []

    for step_key in sorted(process_group.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        if not step_key.isdigit():
            continue

        step_data = process_group[step_key]
        if "loss" in step_data:
            steps.append(int(step_key))
            losses.append(float(step_data["loss"][()]))

            if "grad_norm" in step_data:
                grad_norms.append(float(step_data["grad_norm"][()]))

            if "hessian_eigs" in step_data:
                hessian_eigs.append(np.array(step_data["hessian_eigs"][:]))

    # 处理成numpy数组
    data["steps"] = np.array(steps)
    data["losses"] = np.array(losses)
    if grad_norms:
        data["grad_norms"] = np.array(grad_norms)
    if hessian_eigs:
        data["hessian_eigs"] = np.array(hessian_eigs)

    return data


def plot_training_curves(data, save_path=None, show_plot=True):
    """绘制训练曲线对比图"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Central Flows: Full-batch vs Mini-batch vs Central Flow", fontsize=14)

    # Loss 曲线
    ax_loss = axes[0, 0]
    if "full_batch" in data and "losses" in data["full_batch"]:
        ax_loss.plot(data["full_batch"]["steps"], data["full_batch"]["losses"],
                    label="Full-batch", color="blue", linewidth=2)
    if "mini_batch" in data and "losses" in data["mini_batch"]:
        ax_loss.plot(data["mini_batch"]["steps"], data["mini_batch"]["losses"],
                    label="Mini-batch", color="red", linewidth=2)
    if "central_flow" in data and "losses" in data["central_flow"]:
        ax_loss.plot(data["central_flow"]["steps"], data["central_flow"]["losses"],
                    label="Central Flow", color="green", linewidth=2)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")  # 对数尺度更清楚

    # 梯度范数
    ax_grad = axes[0, 1]
    if "full_batch" in data and "grad_norms" in data["full_batch"]:
        ax_grad.plot(data["full_batch"]["steps"], data["full_batch"]["grad_norms"],
                    label="Full-batch", color="blue", linewidth=2)
    if "mini_batch" in data and "grad_norms" in data["mini_batch"]:
        ax_grad.plot(data["mini_batch"]["steps"], data["mini_batch"]["grad_norms"],
                    label="Mini-batch", color="red", linewidth=2)
    if "central_flow" in data and "grad_norms" in data["central_flow"]:
        ax_grad.plot(data["central_flow"]["steps"], data["central_flow"]["grad_norms"],
                    label="Central Flow", color="green", linewidth=2)

    ax_grad.set_xlabel("Step")
    ax_grad.set_ylabel("Gradient Norm")
    ax_grad.set_title("Gradient Norm")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_yscale("log")

    # Hessian 特征值 (如果有)
    ax_hess = axes[1, 0]
    has_hessian_data = False

    if "full_batch" in data and "hessian_eigs" in data["full_batch"]:
        hess_data = data["full_batch"]["hessian_eigs"]
        if len(hess_data) > 0:
            # 只画前几个特征值
            n_eigs_to_plot = min(5, hess_data.shape[1])
            for i in range(n_eigs_to_plot):
                ax_hess.plot(data["full_batch"]["steps"], hess_data[:, i],
                           label=f"Full-batch λ{i}", linestyle="--", alpha=0.7)
            has_hessian_data = True

    if "mini_batch" in data and "hessian_eigs" in data["mini_batch"]:
        hess_data = data["mini_batch"]["hessian_eigs"]
        if len(hess_data) > 0:
            n_eigs_to_plot = min(5, hess_data.shape[1])
            for i in range(n_eigs_to_plot):
                ax_hess.plot(data["mini_batch"]["steps"], hess_data[:, i],
                           label=f"Mini-batch λ{i}", alpha=0.7)
            has_hessian_data = True

    if "central_flow" in data and "hessian_eigs" in data["central_flow"]:
        hess_data = data["central_flow"]["hessian_eigs"]
        if len(hess_data) > 0:
            n_eigs_to_plot = min(5, hess_data.shape[1])
            for i in range(n_eigs_to_plot):
                ax_hess.plot(data["central_flow"]["steps"], hess_data[:, i],
                           label=f"Central Flow λ{i}", linestyle=":", alpha=0.7)
            has_hessian_data = True

    if has_hessian_data:
        ax_hess.set_xlabel("Step")
        ax_hess.set_ylabel("Hessian Eigenvalue")
        ax_hess.set_title("Top Hessian Eigenvalues")
        ax_hess.legend()
        ax_hess.grid(True, alpha=0.3)
        ax_hess.set_yscale("log")
    else:
        ax_hess.text(0.5, 0.5, "No Hessian data available",
                    transform=ax_hess.transAxes, ha="center", va="center")
        ax_hess.set_title("Hessian Eigenvalues")

    # 参数差异对比 (如果有多个进程)
    ax_diff = axes[1, 1]

    # 这里可以添加更多的统计信息或对比
    ax_diff.text(0.1, 0.8, "Training Comparison:\n\n", fontsize=12, fontweight="bold")

    process_names = []
    if "full_batch" in data and "losses" in data["full_batch"] and len(data["full_batch"]["losses"]) > 0:
        final_loss = data["full_batch"]["losses"][-1]
        process_names.append(f"Full-batch: {final_loss:.4f}")
    if "mini_batch" in data and "losses" in data["mini_batch"] and len(data["mini_batch"]["losses"]) > 0:
        final_loss = data["mini_batch"]["losses"][-1]
        process_names.append(f"Mini-batch: {final_loss:.4f}")
    if "central_flow" in data and "losses" in data["central_flow"] and len(data["central_flow"]["losses"]) > 0:
        final_loss = data["central_flow"]["losses"][-1]
        process_names.append(f"Central Flow: {final_loss:.4f}")

    if not process_names:
        ax_diff.text(0.1, 0.6, "No valid data found", fontsize=10, transform=ax_diff.transAxes)
    else:
        for name in process_names:
            ax_diff.text(0.1, 0.8 - 0.15 * process_names.index(name),
                        name, fontsize=10, transform=ax_diff.transAxes)

    ax_diff.set_xlim(0, 1)
    ax_diff.set_ylim(0, 1)
    ax_diff.axis("off")
    ax_diff.set_title("Training Summary")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Central Flows training curves")
    parser.add_argument("experiment_path", help="Path to experiment folder containing data.hdf5")
    parser.add_argument("--save-path", help="Path to save the plot (optional)")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot (useful for headless)")

    args = parser.parse_args()

    try:
        # 加载数据
        print(f"Loading data from: {args.experiment_path}")
        data = load_experiment_data(args.experiment_path)

        # 检查加载了什么数据
        print("Loaded data for:")
        for key in data:
            if "losses" in data[key] and len(data[key]["losses"]) > 0:
                steps = len(data[key]["steps"])
                final_loss = data[key]["losses"][-1]
                print(f"  - {key}: {steps} steps, final loss: {final_loss:.6f}")
            elif key in data:
                print(f"  - {key}: No valid loss data found")

        # 绘图
        plot_training_curves(data, save_path=args.save_path, show_plot=not args.no_show)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
