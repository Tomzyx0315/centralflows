# CentralFlows Mini-batch 修改说明

## 修改概述

本修改为中央流(Central Flows)项目添加了mini-batch训练支持，允许同时比较full-batch和mini-batch版本的行为差异。

## 主要修改

### 1. 数据集修改 (`src/datasets.py`)
- 添加 `train_batch_size` 参数控制mini-batch大小
- 修改 `Dataset` NamedTuple包含两个training数据集版本
- 新增 `_create_training_batches()` 方法重新组织批次

### 2. 主训练脚本修改 (`main.py`)
- 新增参数：`epochs`, `batch_size`, `compare_full_vs_mini`
- 支持同时运行 `discrete_full` 和 `discrete_mini` 进程
- 确保每次参数更新后立即计算full-batch Hessian

### 3. 公平的对比机制

#### mini-batch版本：
1. 每个epoch遍历所有训练批次
2. **每个batch使用该batch的梯度更新参数**
3. 立即在新的参数位置计算full-batch Hessian

#### full-batch版本：
1. **每个batch也更新一次参数**（与mini-batch更新频率完全相同）
2. **但始终使用全数据集的梯度**
3. 然后计算full-batch Hessian

**核心优势**：两个版本具有完全相同的更新频率和Hessian计算频率，只有梯度来源不同，确保公平对比。

## 使用方法

### 1. 训练对比实验

```bash
# 对比full-batch和mini-batch训练
python main.py opt:gd data:moons arch:mlp \
    --opt.lr=0.02 \
    --epochs=20 \
    --batch-size=32 \
    --compare-full-vs-mini \
    --runs discrete

# 同时对比 full-batch、mini-batch和central flow
python main.py opt:gd data:moons arch:mlp \
    --opt.lr=0.02 \
    --epochs=20 \
    --batch-size=32 \
    --compare-full-vs-mini \
    --runs discrete central
```

### 2. 单独训练mini-batch

```bash
# 只运行mini-batch版本
python main.py opt:gd data:cifar10 arch:cnn \
    --opt.lr=0.01 \
    --epochs=50 \
    --batch-size=128 \
    --compare-full-vs-mini=False
```

## 数据保存格式

训练数据保存为HDF5格式，包含：
- `discrete_full/step_N/`: full-batch版本第N步的数据
- `discrete_mini/step_N/`: mini-batch版本第N步的数据
- `comparison/`: epoch级别的对比统计

## 可视化工具

使用提供的 `plot_training_curves.py` 脚本可视化结果：

使用训练程序输出的实际路径替换下面的 `<experiment_path>`。

例如，如果训练输出显示："Saving data to: experiments/xyz789abc"

```bash
# 可视化训练结果
python plot_training_curves.py experiments/xyz789abc

# 保存图片而不显示GUI
python plot_training_curves.py experiments/xyz789abc --save-path=results.png --no-show
```

可视化图表包括：
1. **Training Loss**: 三个版本的损失曲线对比
2. **Gradient Norm**: 梯度范数演化
3. **Hessian Eigenvalues**: 前几个Hessian特征值的演化
4. **Training Summary**: 最终损失对比

## 理论保证

### Hessian计算原则
- **位置相关**: Hessian在各自历程的参数位置计算
- **实时更新**: 每次参数更新后立即重新计算Hessian
- **理论一致**: Central Flows理论基于full-batch Hessian分析，使用full-batch数据保证理论正确性

### 批次处理:
- mini-batch使用当前batch梯度进行实际参数更新
- Hessian使用full-batch数据确保分析的一致性

## 关键设计决策

1. **Hessian始终基于全数据**: 保证理论框架的正确性
2. **每个更新后立即计算**: 确保Hessian信息的新鲜度
3. **保存完整轨迹**: 允许后续分析不同优化路径的行为差异
4. **公平的更新频率**: 两个版本以相同的切片频率进行更新

## 示例输出

运行对比实验后，您将看到两个版本收敛轨迹的差异，以及它们各自在相同Hessian景观下的行为表现。这对于理解mini-batch SGD相对于full-batch GD的优势和局限性至关重要。

## 实验建议

1. **开始小规模测试**: 先在简单数据集如Moons上测试功能
2. **渐进式对比**: 从相同的batch大小开始，然后尝试不同的batch_size
3. **理论验证**: 观察mini-batch是否能达到类似的sharpness最小值
4. **收敛分析**: 比较两个版本的收敛速度和最终性能

## 核心优势

- **精确对比**: 相同的更新频率和Hessian计算频率
- **理论严谨**: 始终使用full-batch Hessian保持Central Flows理论的适用性
- **实验价值**: 可以隔离梯度估计对优化的影响
