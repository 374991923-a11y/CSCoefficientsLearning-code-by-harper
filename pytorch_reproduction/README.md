# CLOMP 的 PyTorch 复现

对应论文：**Training-Free Ultra Small Model for Universal Sparse Reconstruction in Compressed Sensing**。

本目录是独立、模块化的 PyTorch 实现，保留 `code/` 下的作者原始脚本作为参照。目前复现论文重点分析的 **CLOMP**，包含：

- 1D 合成信号实验，以及 OMP、IHT 基线；
- 2D 灰度图像的按列压缩感知重建；
- 固定测量字典与 DCT 基、可学习稀疏系数；
- 残差相关度归一化和累计支撑集；
- 1D TV、2D TV 与 LV 先验；
- Adam、线性衰减学习率、早停；
- SSIM、PSNR、MSE、PCC 和耗时统计；
- CPU、CUDA、Apple MPS 设备选择与固定随机种子。

这里的“training-free”是指**不使用训练集或预训练权重**。每次重建仍会针对当前测量值执行梯度优化。一个长度为 `n` 的信号只有 `n` 个可学习参数；`batch` 个信号共有 `batch × n` 个参数，测量矩阵和 DCT 基均登记为不可训练 buffer。

## 环境

建议在虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pytorch_reproduction/requirements.txt
```

核心计算只需要 PyTorch、NumPy、SciPy；Pillow 只用于读取 PNG/JPEG/TIFF 图像。无需 OpenCV、PaddlePaddle、torchinfo 或 scikit-image。

## 快速运行

以下是小规模 CPU 冒烟实验，不是论文数值 benchmark：

```bash
python pytorch_reproduction/run_1d.py \
  --n 128 --num-signals 8 --sampling-rate 0.3 \
  --sparsity-rate 0.2 --iterations 30 \
  --methods clomp,iht --device cpu --save-predictions

python pytorch_reproduction/run_2d.py \
  --height 32 --width 32 --sampling-rate 0.3 \
  --iterations 30 --baseline-iterations 30 \
  --methods clomp,iht --device cpu --save-predictions
```

每次运行创建一个新的 `pytorch_reproduction/runs/<实验名>/`，其中包括：

- `summary.json`：完整配置、指标、迭代数、耗时和平均支撑集大小；
- `summary.csv`：便于比较的指标表；
- `losses.json`：总损失和测量 MSE 曲线；
- `predictions.npz`：指定 `--save-predictions` 时的真值与各算法重建结果。

## 论文设置的 1D 实验

论文在每个 `(n, Sr, Kr)` 组合下生成 1000 个信号。下面命令对应作者脚本的典型设置 `n=1000, Sr=0.15, Kr=0.4`：

```bash
python pytorch_reproduction/run_1d.py \
  --n 1000 --num-signals 1000 \
  --sampling-rate 0.15 --sparsity-rate 0.4 \
  --iterations 200 --threshold 0.7 \
  --learning-rate 2 --min-learning-rate 0.01 \
  --tv-weight 0.5 --methods clomp,omp,iht \
  --device auto --log-every 20
```

默认 IHT 稀疏度为 `K = floor(m/5)`，与作者代码一致；可用 `--iht-sparsity` 覆盖。OMP 对每个信号做伪逆，在大批量下会明显慢于 CLOMP。只验证 CLOMP 时可指定 `--methods clomp`。

## 2D 图像实验

使用自己的图像：

```bash
python pytorch_reproduction/run_2d.py \
  --image /path/to/image.tif \
  --sampling-rate 0.3 --iterations 150 \
  --baseline-iterations 200 \
  --learning-rate 2 --min-learning-rate 0.1 \
  --tv-weight 0.2 --lv-weight 0.1 \
  --methods clomp,omp,iht --device auto \
  --save-predictions --log-every 20
```

图像被归一化到 `[0,1]`，高度对应信号长度 `n=H`，宽度对应并行信号数 `W`。测量关系为 `Y=MX`，稀疏基为 DCT，`A=MD`。二维 LV 使用 `3×3` 窗口和步长 2。可用 `.npy` 保存的二维数组代替普通图像文件，也可用 `--height/--width` 同时指定缩放尺寸；不提供图像时使用平滑合成图进行功能验证。

## 算法对应关系

- `cscl/model.py`：论文式 (8)–(10)，残差注入、Mask 和可学习的 `s`；
- `cscl/losses.py`：式 (5)–(7)，TV/LV 先验；
- `cscl/algorithms.py`：CLOMP 优化、OMP 伪逆基线和 IHT；
- `cscl/data.py`：作者 1D 合成信号生成流程；
- `cscl/metrics.py`：1D/2D 评价指标；
- `run_1d.py` / `run_2d.py`：可复现实验入口。

相较原始脚本，修正了未定义的 1D 学习率调度器、CUDA 硬编码、CPU 上调用 `.numpy()`、支撑集广播和归一化字典系数的反变换等问题。重建前先撤销字典原子归一化尺度，再计算 `x=Ds`。最终指标对输出裁剪到 `[0,1]`。

### 先验模式：论文公式 vs 官方代码

论文式 (4) 与 Fig. 2b 将 TV/LV 先验和最终输出都作用在 `Mask⊙s` 上；而作者发布脚本中，先验损失与最终输出使用 **Mask 之前的全部系数**（Mask 只约束测量分支），等价于稠密 TV 正则化重建。本复现默认论文公式行为，加 `--unmasked-prior` 切换为官方代码行为。实测（n=128, 200 迭代，Sr=0.3, Kr=0.2）：两者 SSIM 0.915 vs 0.924，收敛后差距很小；未充分收敛时官方行为在该设置下反而更低（30 迭代 0.263 vs 0.472）。

注意 CLOMP 不是无条件优于 OMP：论文优势区是低采样率 + 高稀疏度。温和设置（如 Sr=0.3, Kr=0.2）下 OMP 已接近完美恢复，论文中该区域 CLOMP≈OMP；且 CLOMP 需 100–200 迭代收敛（`runs/diag_1d_*` 保留了完整的 2×2 诊断与 n=1000/Sr=0.1/Kr=0.4 优势区验证）。

## 测试

无需 pytest：

```bash
python -m unittest discover -s pytorch_reproduction/tests -v
```

测试覆盖 DCT 正交性、数据确定性、归一化字典一致性、参数量、支撑集累计、梯度下降、OMP/IHT 简单精确恢复、先验可微性和指标。

## 复现边界

- 根目录 `Data.7z` 当前是 Git LFS 指针，不是约 338 MB 的实际数据。因此默认命令不依赖该数据，也不宣称复现论文表格中的全部数值。
- 当前范围是论文的 CLOMP 1D/2D 核心实验。论文另述的 CLMP、CLISTA、CLAMP、Deep Unfolding 对照及九个图像数据集实验尚未包含。
- 论文没有列出所有图像预处理与逐数据集超参数；不同硬件、库版本、随机种子和早停路径也会影响结果。复现论文图表时应固定这些条件并逐个组合运行。
