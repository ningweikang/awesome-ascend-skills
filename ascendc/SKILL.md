---
name: ascendc
description: Guides the agent to develop AscendC transformer GMM-style custom ops (such as grouped_matmul_finalize_routing) and their CANN aclnn examples by following existing patterns under ops-transformer/gmm and attention/softmax_ops/examples. Use when adding or modifying these ops, their kernels, tiling/infershape logic, or CANN API examples.
---

# AscendC Transformer 算子开发

本技能指导 agent 按现有模式开发/修改 AscendC transformer 相关算子，包括：
- FFN (Feed Forward Network) 算子
- GMM (Grouped Matrix Multiplication) 类算子
- MoE (Mixture of Experts) 路由类算子
以及对应的 CANN `aclnn_*` 示例代码。

## 使用时机

在以下场景应用本技能：

- 需要新增或修改 FFN (Feed Forward Network) 相关的 AscendC 算子
- 需要新增或修改 GMM (Grouped Matrix Multiplication) 类 AscendC 算子
- 需要新增或修改 MoE (Mixture-of-Experts) 路由类 AscendC 算子
- 需要为已有 AscendC 算子补充 `op_host` 定义、tiling / infershape 或 `op_kernel` 实现
- 需要编写类似 `ffn/ffn/examples/test_aclnn_ffn.cpp` 的 CANN `aclnn_*` 示例
- 需要对这些算子进行对齐、重构或 bug 修复，同时保持与现有算子风格一致

---

## 总体工作流

当用户要求开发/修改此类算子时，按下面步骤执行（顺序很重要）：

1. **定位参考算子/示例**
   - 根据算子类型，在相应目录下查找：
     - FFN 算子：`ops-transformer/ffn/`
     - GMM 算子：`ops-transformer/gmm/`
     - MoE 算子：`ops-transformer/moe/`
   - 查找以下类型文件：
     - `*_def.cpp`（算子定义）
     - `*_tiling*.h/.cpp`（tiling、调度逻辑）
     - `op_kernel/*.h`（AscendC kernel 实现）
   - 在对应算子目录下的 `examples/` 子目录中查找 CANN `aclnn_*` 示例，例如：
     - FFN: `ffn/ffn/examples/test_aclnn_ffn.cpp`
     - GMM: `gmm/grouped_matmul/examples/`
     - MoE: `moe/moe_init_routing/examples/`

2. **在 op_host 中定义 Graph 算子接口**
3. **在 op_kernel 中实现 AscendC kernel（含量化/路由逻辑）**
4. **补齐/复用 tiling、infershape 与注册逻辑（如果相关文件存在）**
5. **编写或更新 CANN API 示例与单测**

后续各节会具体说明每一步要做什么、关注哪些细节。

---

## 步骤一：复用现有模式

### 必读参考

- FFN 算子参考：
  - Graph 定义：`ops-transformer/ffn/ffn/op_host/ffn_def.cpp`
  - Tiling 实现：`ops-transformer/ffn/ffn/op_host/ffn_tiling.cpp`
  - CANN API 示例：`ops-transformer/ffn/ffn/examples/test_aclnn_ffn.cpp`

- GMM 算子参考：
  - Graph 定义：`ops-transformer/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp`
  - AscendC kernel 实现：`ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul.h`

- MoE 算子参考：
  - Graph 定义：`ops-transformer/moe/moe_init_routing/op_host/moe_init_routing_def.cpp`
  - CANN API 示例：`ops-transformer/moe/moe_init_routing/examples/test_aclnn_moe_init_routing.cpp`

- 通用 CANN API 示例参考：
  - `ops-transformer/attention/softmax_ops/examples/test_aclnn_softmax_ops.cpp`

### 行为规范

- **永远从现有同类算子拷贝骨架，再做最小必要修改**
- 保持：
  - 命名风格（文件名、类名、命名空间）
  - 宏使用方式（`ASCEND_IS_AIC`、`ASCEND_IS_AIV` 等）
  - 队列、UB buffer 管理模式（`TQue`、`TPipe`）
  - AICore 配置与不同芯片支持方式（例如 `ascend910b` / `ascend910_95`）

---

## 步骤二：在 op_host 中定义算子接口

### 关键模式

继承 `OpDef`，在 `namespace ops` 内定义类，并使用 `OP_ADD` 注册：
  - 输入：
    - 使用 `Input("name")` + `.ParamType(REQUIRED/OPTIONAL)`
    - 明确 `.DataType({ ... })`、`.Format({ ... })` 与 `.UnknownShapeFormat({ ... })`
    - 支持多场景/多类型时，用向量形式列出所有组合
  - 输出：
    - 使用 `Output("y")`，同样配置 DataType / Format
  - 属性：
    - 用 `.Attr("attr_name").AttrType(OPTIONAL/REQUIRED).Int/Float/Bool/ListInt(...)` 设置默认值
  - AICore 配置：
    - 构造 `OpAICoreConfig`，设置：
      - `DynamicCompileStaticFlag(true)`
      - `DynamicFormatFlag(true)`
      - `DynamicRankSupportFlag(true)`
      - `DynamicShapeSupportFlag(true)`
      - `NeedCheckSupportFlag(false)`（如参考算子这么写）
      - 必要的 `ExtendCfgInfo(...)`，例如 `"softsync.flag"`, `"prebuildPattern.value"`, `"coreType.value"`, `"aclnnSupport.value"`
    - 按芯片型号调用 `this->AICore().AddConfig("ascend910b", config);` 等
  - 末尾用 `OP_ADD(YourOpClassName);` 完成注册

### 算子特定示例

#### FFN 算子（参考 `ffn_def.cpp`）

FFN 算子支持 Feed Forward Network 计算，可选多种激活函数：

```cpp
// 输入定义
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight1")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight2")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias1")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});
// 输出定义
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});

// 属性定义
Attr("activation").AttrType(OPTIONAL).Int({0}); // 0: GELU, 1: RELU, 2: FASTGELU, 3: SILU, 4: SIGMOID, 5: TANH
Attr("inner_precise").AttrType(OPTIONAL).Int({0}); // 0: BF16, 1: FLOAT32
```

#### GMM 算子（参考 `grouped_matmul_def.cpp`）

GMM 算子支持分组矩阵乘法，可配置分组方式和数据类型：

```cpp
// 输入定义
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});

// 输出定义
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32, DT_INT8})
    .Format({FORMAT_ND});

// 属性定义
Attr("split_item").AttrType(OPTIONAL).ListInt({}); // 分组信息
Attr("dtype").AttrType(OPTIONAL).Int({0}); // 0: FLOAT16, 1: BF16, 2: INT8
Attr("transpose_weight").AttrType(OPTIONAL).Int({0}); // 0: 不转置, 1: 转置
```

#### MoE 算子（参考 `moe_init_routing_def.cpp`）

MoE 算子支持 Mixture-of-Experts 路由逻辑：

```cpp
// 输入定义
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Input("rowIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Input("expertIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// 输出定义
Output("expandedXOut")
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Output("expandedRowIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Output("expandedExpertIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// 属性定义
Attr("activeNum").AttrType(OPTIONAL).Int({0}); // 激活的专家数量
```

### Agent 要点

- 为新算子时：
  - **从参考算子完整复制类声明和构造函数体**，然后只改：
    - 类名 / 文件名
    - 输入输出名称与数量
    - 支持的 `DataType` / `Format`
    - 特有属性与默认值
  - 如无特殊理由，不要随意改动参考算子已有的 AICore flag 与 `ExtendCfgInfo` 结构
- 如果需要支持 `aclnn`：
  - 仿照参考算子中的 `"aclnnSupport.value", "support_aclnn"` 配置

---

## 步骤三：在 op_kernel 中实现 AscendC kernel

### 通用特征

- 使用与算子同名的命名空间（如 `namespace FFN`, `namespace GroupedMatmul`, `namespace MoeInitRouting`）
- 引入必要头文件：
  - `kernel_operator.h`
  - `lib/matmul_intf.h`（矩阵乘相关算子）
  - 自己的工具头文件（如 `ffn_utils.h`, `grouped_matmul_utils.h`）
- 定义类型别名：
  - `using aT = MatmulType<...>;`
  - `using bT = MatmulType<...>;`
  - `using BiasT = ...;`
  - `using cT = ...;`
  - `using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;`
- 使用模板参数控制不同场景（数据类型、量化模式、激活函数等）

### 算子特定实现

#### FFN 算子实现

FFN 算子实现 Feed Forward Network 计算，包含两个线性变换和激活函数：

```cpp
namespace FFN {

// 定义激活类型枚举
enum ActiveType {
    ACTIVE_GELU = 0,
    ACTIVE_RELU = 1,
    ACTIVE_FASTGELU = 2,
    ACTIVE_SILU = 3,
    ACTIVE_SIGMOID = 4,
    ACTIVE_TANH = 5
};

// 定义参数结构体
template <typename T, ActiveType ACTIVE, bool WITH_BIAS>
struct Param {
    using InputType = T;
    using OutputType = T;
    static constexpr ActiveType kActive = ACTIVE;
    static constexpr bool kWithBias = WITH_BIAS;
};

// 主计算类
template <class P> class FfnCompute {
public:
    using InputType = typename P::InputType;
    using OutputType = typename P::OutputType;

    // 初始化函数
    void Init(const InitParams &initParams, const FFNTiling *tiling) {
        // 初始化全局张量、UB buffer、队列等
    }

    // 处理函数
    void Process() {
        // 第一个线性变换：x * weight1 + bias1
        // 应用激活函数
        // 第二个线性变换：(x * weight1 + bias1) * weight2 + bias2
        // 写回结果
    }

private:
    // 实现激活函数
    void ApplyActivation(InputType *src, OutputType *dst, uint32_t size) {
        switch (P::kActive) {
            case ACTIVE_GELU:
                // 实现 GELU 激活
                break;
            case ACTIVE_FASTGELU:
                // 实现 FASTGELU 激活
                break;
            // 其他激活函数实现
        }
    }
};

} // namespace FFN
```

#### GMM 算子实现

GMM 算子实现分组矩阵乘法：

```cpp
namespace GroupedMatmul {

// 定义参数结构体
template <typename T, typename WeightT, typename BiasT, typename OutputT>
struct Param {
    using InputType = T;
    using WeightType = WeightT;
    using BiasType = BiasT;
    using OutputType = OutputT;
};

// 主计算类
template <class P> class GroupedMatmulCompute {
public:
    using InputType = typename P::InputType;
    using WeightType = typename P::WeightType;
    using BiasType = typename P::BiasType;
    using OutputType = typename P::OutputType;

    // 初始化函数
    void Init(const InitParams &initParams, const GroupedMatmulTiling *tiling) {
        // 初始化全局张量、分组信息、UB buffer、队列等
    }

    // 处理函数
    void Process() {
        // 循环处理每个分组
        for (uint32_t groupIdx = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
            // 计算当前分组的矩阵乘
            ComputeGroup(groupIdx);
        }
    }

private:
    // 分组计算函数
    void ComputeGroup(uint32_t groupIdx) {
        // 设置当前分组的输入、权重、输出偏移
        // 执行矩阵乘法
        // 添加偏置（如果有）
        // 写回当前分组结果
    }
};

} // namespace GroupedMatmul
```

#### MoE 算子实现

MoE 算子实现 Mixture-of-Experts 路由逻辑：

```cpp
namespace MoeInitRouting {

// 定义参数结构体
template <typename T, typename IndexT>
struct Param {
    using InputType = T;
    using IndexType = IndexT;
};

// 主计算类
template <class P> class MoeInitRoutingCompute {
public:
    using InputType = typename P::InputType;
    using IndexType = typename P::IndexType;

    // 初始化函数
    void Init(const InitParams &initParams, const MoeInitRoutingTiling *tiling) {
        // 初始化全局张量、UB buffer、队列等
    }

    // 处理函数
    void Process() {
        // 处理路由逻辑
        // 根据 rowIdx 和 expertIdx 扩展输入 x
        // 生成扩展后的 rowIdx 和 expertIdx
        // 写回结果
    }

private:
    // 扩展输入张量
    void ExpandInput(const InputType *x, IndexType *rowIdx, IndexType *expertIdx,
                    InputType *expandedX, IndexType *expandedRowIdx, IndexType *expandedExpertIdx) {
        // 实现扩展逻辑
    }
};

} // namespace MoeInitRouting
```

### 典型结构（参考即可，勿死记）

- 工具函数：
  - 如 `DataCopyPad2D`，有 GM↔UB 两个重载，带 `DataCopy2DDimParams`
- 主要类：
  - 包含 `Init(...)` 方法：初始化全局张量、UB buffer、队列等
  - 包含 `Process()` 方法：整体执行流程，包含计算逻辑和结果写回
  - 包含私有辅助方法：实现具体计算逻辑（如激活函数、分组处理等）
  - 若新算子逻辑相近，**尽量复用这一整套结构，只做必要变更**

### Agent 要点

- 为新算子/变体时：
  - 先确认：
    - 是否仍基于 `MatmulImpl`，以及需要哪些 GM tensor
    - tiling 结构里有哪些字段（例如 `matmulTiling.baseM/baseN/k`、`groupNum` 等）
  - 修改点仅限：
    - 新增/删减 GM 输入（例如多了一个 scale/bias/logits）
    - 调整 `ComputeDequantAndActivate` / `PerTokenScaleBrcb` 等中使用的张量组合
    - 修改 `InitOutputWithZeros`、`PreProcess` 中与业务强相关的初始化逻辑
  - 保持：
    - 队列/UB 分配、`PipeBarrier`、`DataCopyPad`、`SetAtomicAdd` 这些模式不变，除非有明确 bug 或需求

---

## 步骤四：tiling / infershape / 其他 host 逻辑

虽然本技能示例未展开全部文件，但 agent 在代码库中应遵循以下模式：

1. 在 `op_host/` 下查找：
   - `*_tiling*.h/.cpp`
   - `*_infershape.cpp`
   - 其他 `${op_name}_*.cpp` 文件
2. 分析参考算子中的：
   - tiling 入参（batch、M/N/K、group 数、是否 deterministic 等）
   - 如何把 Graph 级别 shape/attr 转为 kernel 所需的 `tiling` 结构
3. 为新算子时：
   - 若语义接近，优先复制参考 tiling/infershape 代码后改名、改字段
   - 确保：
     - Graph 中的 attr / shape 能正确映射到 kernel 中访问的 `tiling->...` 字段
     - deterministic 开关、workspace size、coreNum / parallNum 等计算逻辑保持一致风格

---

## 步骤五：CANN aclnn 示例（examples）

参考 `test_aclnn_softmax_ops.cpp`，模式如下：

1. **通用工具函数**
   - `GetShapeSize`：计算形状乘积
   - `PrintOutResult<T>`：将 device 结果拷回 host 并打印
   - `Init(deviceId, &stream)`：
     - `aclInit`
     - `aclrtSetDevice`
     - `aclrtCreateStream`
   - `CreateAclTensor<T>`：
     - 使用 `aclrtMalloc` 分配 device 内存
     - `aclrtMemcpy` 从 host 拷贝到 device
     - 计算连续 tensor 的 `strides`
     - 调用 `aclCreateTensor` 创建 `aclTensor*`

2. **主流程 main()**
   - 初始化 ACL runtime
   - 为每个输入/输出构造：
     - host 侧数据（`std::vector<T>`）
     - 对应的 shape（`std::vector<int64_t>`）
     - 调用 `CreateAclTensor(...)` 构造 `aclTensor*` 和 device addr
   - 获取 workspace 大小与执行器：
     - 调用 `aclnnYourOpGetWorkspaceSize(...)`
   - 如 `workspaceSize > 0`：
     - `aclrtMalloc` 分配 workspace
   - 调用真正算子：
     - `aclnnYourOp(workspaceAddr, workspaceSize, executor, stream);`
   - `aclrtSynchronizeStream(stream);`
   - 拷回输出并调用 `PrintOutResult` 打印结果
   - 销毁 tensor、释放 device 内存、销毁 stream、reset device、`aclFinalize`

### Agent 要点

- 为新 `aclnn_*` 示例时：
  - 完整 copy `test_aclnn_softmax_ops.cpp` 的结构，修改：
    - 头文件 include（`aclnnop/aclnn_xxx.h`）
    - 张量个数、shape、dtype 和填充数据
    - `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 的函数名和参数列表
  - 保持：
    - 错误检查宏（`CHECK_RET`）和日志输出宏
    - 所有 `acl*` 资源的成对申请/释放

---

## 步骤六：测试与验证（如有 Python 前端）

如果项目中存在 Python 测试（例如 `op-plugin/test/test_custom_ops/test_*.py`）：

1. 使用现有测试文件作为模板：
   - 名称一般为 `test_npu_<op_name>_*.py`
   - 使用 NPU 设备，调用前端 API 封装的 op
2. 为新算子时：
   - 构造典型输入形状（包括边界场景）
   - 若有可对标的参考实现（如 CPU 版本或简单 Python 算法），用它计算期望输出
   - 断言：
     - shape、dtype 正确
     - 数值误差在合理范围内（尤其是量化/反量化场景）

---

## 对 agent 的额外约束

- **不要偷懒从零发明模式**：总是先查找并对齐相邻算子的实现与示例
- 修改任何已存在文件前：
  - 先完整阅读该文件，理解其在整体算子中的角色
- 对于涉及芯片支持 / 动态 shape / deterministic 的逻辑：
  - 优先保持现有算子的一致性，除非 bug 或需求要求改变
- 对示例和测试：
  - 宁可示例小而清晰（单个 shape、易于人工检查），也不要一开始就做过多复杂场景

---

## 简要使用示例

- **用户**：新增一个和 `grouped_matmul_finalize_routing` 类似，但 scale/bias 组合不同的 GMM 路由算子
- **Agent 行为（遵循本技能）**：
  1. 在 `gmm` 目录下找到 `grouped_matmul_finalize_routing_*` 相关文件并阅读
  2. 复制 `*_def.cpp`，改名和接口，调整输入/属性
  3. 复制 `op_kernel/grouped_matmul_finalize_routing.h` 中核心类，根据新需求调整 GM tensor 和反量化流程
  4. 参考相关 tiling/infershape 文件，确保 Graph 到 kernel 的参数映射正确
  5. 参考 `test_aclnn_softmax_ops.cpp` 写一个新的 `aclnn` 示例，并在需要时补充单测

