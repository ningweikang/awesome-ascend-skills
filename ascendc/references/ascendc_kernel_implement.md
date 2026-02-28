# AscendC Kernel 开发参考

本文档用于帮助识别和避免 AscendC 核函数开发中的常见问题，并给出合理目标与正确模式。文档会持续补全。

---

## 目标（Kernel 开发时应达到的共识）

- **正确区分 GlobalTensor 与 TQue 的职责**：GM 读写用 GlobalTensor + DataCopy；双缓冲流水用 TQue（AllocTensor / EnQue / DeQue / FreeTensor）。
- **Init 阶段**：对 **TQue** 调用 `pipe.InitBuffer(queue, BUFFER_NUM, size)`，对 **GlobalTensor** 只做 `SetGlobalBuffer(ptr, length)`。
- **CopyIn / Compute / CopyOut**：队列相关操作只作用在 TQue 上；与 GM 的数据交换只通过 `GlobalTensor[offset]` 与 `DataCopy` 完成。

---

## 1. GlobalTensor 与 TQue 的职责区分

### 问题识别

若编译报错类似：

- `no member named 'AllocTensor' in 'AscendC::GlobalTensor<...>'`
- `no member named 'EnQue' in 'AscendC::GlobalTensor<...>'`
- `no member named 'DeQue' in 'AscendC::GlobalTensor<...>'`
- `no member named 'FreeTensor' in 'AscendC::GlobalTensor<...>'`

说明把 **TQue 的接口用在了 GlobalTensor 上**，需要改为“队列用 TQue、GM 用 GlobalTensor”。

### 正确职责

| 类型 | 职责 | 常用接口 |
|------|------|----------|
| **GlobalTensor\<T\>** | 表示全局内存上的张量，用于与 GM 之间的读写 | `SetGlobalBuffer(ptr, length)`、`tensor[offset]`（与 DataCopy 配合） |
| **TQue\<QuePosition, BUFFER_NUM\>** | 双缓冲流水队列，管理 LocalTensor 的分配与传递 | `AllocTensor<T>()`、`EnQue(local)`、`DeQue<T>()`、`FreeTensor(local)` |

**GlobalTensor 没有** AllocTensor / EnQue / DeQue / FreeTensor；**TQue 不直接表示 GM 地址**，需与 GlobalTensor 配合完成 GM ↔ 片上数据交换。

### 正确流程（CopyIn / Compute / CopyOut）

1. **CopyIn**
   - 从 **TQue** 上 `AllocTensor<T>()` 得到 LocalTensor。
   - `DataCopy(local, inputGM[progress * tileLength], tileLength)`（GM → local）。
   - 对**输入 TQue** 调用 `EnQue(local)`。

2. **Compute**
   - 从**输入 TQue** 上 `DeQue<T>()` 取 LocalTensor。
   - 从**输出 TQue** 上 `AllocTensor<T>()` 得到输出 LocalTensor。
   - 在 LocalTensor 上做计算，然后对**输出 TQue** 调用 `EnQue(local)`。
   - 对已用完的输入 LocalTensor 调用**输入 TQue** 的 `FreeTensor(local)`。

3. **CopyOut**
   - 从**输出 TQue** 上 `DeQue<T>()` 取 LocalTensor。
   - `DataCopy(outputGM[progress * tileLength], local, tileLength)`（local → GM，目标在前）。
   - 对**输出 TQue** 调用 `FreeTensor(local)`。

### Init 中的正确用法

- **GlobalTensor**：仅设置 GM 视图，例如
  `inputGM.SetGlobalBuffer((__gm__ T*)ptr + blockOffset, blockLength);`
- **TQue**：用 pipe 初始化缓冲区，例如
  `pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(T));`
  不要对 GlobalTensor 调用 `InitBuffer`。

### 参考示例

- 本仓库内可参考：`xpu_kernel/C_like/example/npu/softmax_ops/op_kernel/softmax_ops.h`
  - 成员：`TQue<> inputQueueX, outputQueueS` 与 `GlobalTensor<T> inputGMX, outputGMS`
  - CopyIn：对 queue 做 AllocTensor → DataCopy(GM→local) → EnQue
  - Compute：DeQue → 计算 → 对输出 queue AllocTensor → 写入 → EnQue，并对输入 queue FreeTensor
  - CopyOut：DeQue → DataCopy(local→GM) → FreeTensor

---

## 2. 其他常见问题（待补全）

- （后续可在此增加：tile 长度与对齐、多数据类型分派、Reduce 临时缓冲等）
