---
name: profiling-computing
description: Skill for analyzing computing performance bottlenecks, focusing on operator efficiency and computation optimization in Ascend NPU systems.
---

# Profiling 计算瓶颈分析 Skill

## 功能概述

该Skill用于分析系统中的计算瓶颈问题，当主分析Skill检测到计算耗时占比超过85%时自动触发。

## 分析内容

- **算子性能分析**：识别计算效率低下的算子
- **计算资源利用**：分析NPU计算资源的使用情况
- **计算任务分布**：分析计算任务在不同设备上的分布

## 输出结果

- 算子性能排名报告
- 计算资源利用率分析
- 算子优化建议

## 使用方式

该Skill通常由主分析Skill `/profiling-main` 自动调用，也可以单独使用：

```python
# 运行计算瓶颈分析
python scripts/analyze_computing.py --input <path_to_csv_files>
```