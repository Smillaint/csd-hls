# csd-hls

Federated learning edge-acceleration HLS component for gradient importance scoring.

该仓库包含一个面向联邦学习边缘节点的 Vivado HLS 组件，用于在本地客户端侧评估样本梯度与全局梯度的相关性，并选出更重要的样本索引。核心模块通过余弦相似度计算每个样本梯度对当前全局更新方向的贡献度，可作为边缘端缓存筛选、样本选择或通信前预处理的一部分。

## 功能概述

在联邦学习中，边缘设备通常需要在受限算力、内存和通信带宽下处理本地训练数据。本组件接收一批样本梯度和一个全局梯度向量，计算每个样本梯度与全局梯度之间的余弦相似度：

```text
score_i = dot(sample_grad_i, global_grad)
          / (||sample_grad_i||_2 * ||global_grad||_2)
```

分数越高，表示该样本梯度方向与全局模型更新方向越一致。测试程序会按照分数从高到低排序，并输出 Top-K 样本的真实数据索引到 `selected_indices.txt`。

## HLS 核心

顶层函数：

```cpp
void grad_importance(fixed_t sample_grads[][GRAD_DIM],
                     fixed_t global_grad[GRAD_DIM],
                     fixed_t scores[MAX_SAMPLES],
                     int n_samples);
```

主要参数定义在 `grad_importance.h`：

```cpp
#define GRAD_DIM    80202
#define MAX_SAMPLES 100
#define TOP_K       20
typedef ap_fixed<32, 4> fixed_t;
```

当前配置对应 SimpleCNN 的 80202 维参数梯度，最多一次处理 100 个样本，并默认选取 Top-20。数值类型使用 `ap_fixed<32, 4>`，在精度和 FPGA 资源开销之间做折中。

## 数据流

1. 读取本地样本梯度文件 `grad_client0.txt`。
2. 读取全局梯度文件 `global_grad.txt`。
3. 对每个样本计算 L2 范数、点积和余弦相似度。
4. 按相似度分数降序排序。
5. 输出 Top-K 的真实样本 ID 到 `selected_indices.txt`。

HLS 顶层函数只负责并行友好的分数计算；文件解析、排序和输出逻辑位于 `tb_grad_importance.cpp`，主要用于 C 仿真和系统联调。

## 接口与综合目标

`grad_importance.cpp` 使用 AXI 接口约束：

```cpp
#pragma HLS INTERFACE m_axi port=sample_grads
#pragma HLS INTERFACE m_axi port=global_grad
#pragma HLS INTERFACE m_axi port=scores
#pragma HLS INTERFACE s_axilite port=n_samples
#pragma HLS INTERFACE s_axilite port=return
```

循环内部对梯度维度计算使用 `PIPELINE II=1`，用于提高点积和范数计算吞吐。当前 HLS 脚本目标器件为：

```text
xc7z020clg484-1
clock period: 10 ns
top function: grad_importance
```

## 仓库结构

```text
grad_importance.cpp      HLS 顶层实现，包含 L2 范数、点积和余弦相似度计算
grad_importance.h        参数、定点类型和顶层函数声明
tb_grad_importance.cpp   C 仿真测试程序，负责读入梯度、排序并输出选择结果
solution1/script.tcl     Vivado HLS 工程脚本
solution1/directives.tcl HLS directive 文件
```

生成目录、仿真输出、综合输出和日志文件已通过 `.gitignore` 排除。

## 输入文件格式

测试程序默认从当前目录或 `data_path.cfg` 指定目录读取：

```text
grad_client0.txt
global_grad.txt
```

`grad_client0.txt` 使用轮次和样本索引组织：

```text
Round 1
Initial ...
Data index 0
0.001,0.002,-0.003,...
Data index 1
...
```

`global_grad.txt` 使用轮次组织：

```text
Round 1
0.0005,-0.0012,0.0031,...
```

如果指定轮次不存在，测试程序会选择文件中可用的最高轮次。

## 运行方式

在 Vivado HLS 2019.x 环境中，可使用工程脚本执行 C 仿真、综合、协同仿真和 IP 导出：

```tcl
vivado_hls -f solution1/script.tcl
```

测试程序也支持命令行参数：

```text
tb_grad_importance <grad_file> <global_grad_file> <round_num> <top_k> <max_samples>
```

示例：

```text
tb_grad_importance grad_client0.txt global_grad.txt 1 20 100
```

## 输出

成功运行后会在本地梯度文件所在目录生成：

```text
selected_indices.txt
```

内容格式：

```text
Round 1
12
5
43
...
```

每一行是被选中的真实样本 ID，可回传给上层联邦学习调度逻辑，用于后续缓存更新、样本重放或边缘端训练数据选择。

## 设计定位

该 HLS 组件适合部署在联邦学习边缘加速链路中，承担梯度重要性评分这一计算密集步骤。当前实现重点放在功能正确性和 HLS 可综合性上，后续可继续优化方向包括：

- 对 `GRAD_DIM` 维度循环做分块和并行展开。
- 将样本梯度读取改为流式接口，降低片上缓存压力。
- 使用近似平方根或归一化预处理减少 `sqrt` 成本。
- 将 Top-K 排序逻辑下沉到硬件模块，减少 PS 侧后处理开销。
