# AICAS2026 Profiling & 优化思路（算子级）

## 目标
在不修改 prompt、不做投机采样、不做剪枝/蒸馏/量化/改结构/改权重的前提下，基于 profiling 找到瓶颈，围绕算子与内核级优化提升 TTFT 与 Throughput，同时保持准确率。

## Baseline 产出规则
- 每次 benchmark 结果输出到 `result/mm-dd-hh-mm/` 目录
- 以同样参数进行可复现对比

## Profiling 计划
### 1. 基线性能确认（无任何改动）
- 运行 `benchmark.py` 拿到 TTFT/吞吐量与准确率
- 记录硬件信息与运行参数

### 2. 端到端 Profile（低样本）
目的：定位最占时、最耗显存的模块和算子。
- 使用 `torch.profiler`，开启 CPU/CUDA、shape、memory 统计
- 输出 chrome trace 与 table
- 在 `evaluation_wrapper.py` 内为关键阶段加 NVTX ranges，便于 Nsight Systems/Compute 观察
  - 视觉编码（vision encoder）
  - 文本编码与跨模态融合（cross-modal connector）
  - 生成阶段 prefill 与 decode

### 3. 算子级分析
- 关注热点：Attention、LayerNorm、MLP（GEMM+GELU）、视觉 patch/conv 相关
- 统计每类算子占比、频次、shape 分布
- 重点关注 TTFT 相关路径（prefill 阶段）

### 4. 生成阶段分析
- 关注 KV Cache 的分配/碎片/访问模式
- 每 step decode kernel 数量与同步点
- 是否有不必要的 CPU-GPU 同步

## 优化方向（基于 profile）
### A. Attention/MLP 算子替换与融合
- 优先启用/验证 Flash Attention 或高效注意力实现
- 使用 `scaled_dot_product_attention`（如可用）减少中间张量
- 通过 Triton/自定义 CUDA kernel 融合 QKV、softmax、dropout（若存在）
- 尝试融合 LayerNorm + Linear、GELU + Linear

### B. Vision Encoder 加速
- 针对视觉编码路径的 attention / conv / layernorm 热点做替换
- 检查是否存在不必要的 CPU preprocessing 或同步
- 对固定输入分辨率的路径进行 kernel 选择与缓存

### C. KV Cache 管理优化
- 预分配 KV Cache，使用连续内存布局
- 减少频繁的重新分配与碎片
- 如果可行，启用 page/block KV cache

### D. 编译与图优化
- 对稳定路径（vision encoder / cross-modal connector / text prefill）尝试 `torch.compile`
- 对小 batch 且 shape 稳定的部分尝试 CUDA Graphs

### E. 数值与后端设置（不改权重）
- 允许 TF32（若符合规则）：
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision('high')` 以提升 GEMM 性能

## 迭代流程
1. 基线 → 端到端 profile
2. 按热点优先级实现 1-2 个算子优化
3. 小样本验证正确性 + 速度变化
4. 逐步扩展到 100/5000 样本
5. 记录每次改动的结果与对比结论

## 约束确认
- 不修改 `benchmark.py`
- 只在 `evaluation_wrapper.py` 中实现优化
- 不做 prompt 修改、投机采样、剪枝、蒸馏、量化、改结构/改权重
- 不调用外部服务
