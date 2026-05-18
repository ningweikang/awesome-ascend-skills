# Awesome Ascend Skills 仓库规划设计

## Problem

当前仓库已经从“少量独立 skills 集合”演变为“包含单 skill、bundle、路由 skill、嵌套子 skill、外部同步 skill”的综合知识库，但入口设计、安装模型和信息架构仍偏向早期阶段，导致以下问题逐步放大：

1. 新同学进入仓库后，面对大量 skills，难以快速判断“我该安装哪些”“我该从哪里开始”。
2. README 主要仍是平铺式展示 skill 列表，与仓库当前已经存在的 bundle / skill set 结构不完全匹配。
3. 直接通过 `npx skills add` 安装时，默认暴露的 skill 过多，增加选择负担，也不利于按任务场景进行最小化安装。
4. 随着 skills 持续增长，仓库还面临分类、命名、去重、质量标准和数据反馈机制不足的问题，后续维护成本会持续上升。

## Findings

### 1. Onboarding / Discovery

- 新人首次接触时的核心痛点是：缺少清晰入口、skills 过多、场景映射不清、安装策略不透明、内置与外部 skill 边界感弱。
- 最应该优先补的不是继续增加零散说明，而是建立 **task-based entry**：先告诉用户“你要解决什么问题”，再映射到 skill 与安装方案。

### 2. Packaging / Install Model

- 更适合当前仓库阶段的方案不是继续平铺暴露全部 skills，而是采用 **分层 bundle**。
- bundle 设计应贴近真实工作场景，优先考虑：
  - 基础环境
  - 推理
  - 训练
  - Profiling / 性能分析采集
  - 专项方向（如 AI for Science）
- 后续安装模型应围绕“按层安装、按需暴露、减少默认暴露面、按任务选择 bundle”设计。

### 3. README / Information Architecture

- README 首页应优先解决用户理解路径，而不是优先展示完整清单。
- 信息优先级建议为：
  1. Quick Start Paths
  2. Skill Decision Tree
  3. Catalog by Scenario
  4. Bundle Table
- external boundary、install matrix 等内容可以后置，不应占据首页主要注意力。

### 4. Governance / Maintenance

- 随着数量增长，仓库需要先建立几项基础治理能力：
  1. **taxonomy**：统一分类体系
  2. **quality bar**：统一质量门槛
  3. **analytics**：使用与维护数据反馈
- 这些机制将支持后续的去重、整合、升级和淘汰决策。

### 5. Roadmap / Metrics

- 用户偏好的落地节奏为：
  1. Phase 1：先建设 bundle 体系
  2. Phase 2：重构 README，并给出推荐安装方式
  3. Phase 3：建设治理机制，并观察效果指标
  4. Phase 4：建设 GitHub Pages 等更直观的选择入口
- 成功指标建议围绕：
  - bundle adoption
  - faster selection
  - readme navigation
  - docs discovery

## Recommendation

建议将本仓库从“skill 平铺仓库”升级为“**按场景导航、按层安装、按规则治理**”的 skills 平台，并按以下顺序推进：

### Phase 1：建立分层 bundle 体系

- 定义官方推荐安装层级：如 `base`、`inference`、`training`、`profiling`、`specialized`。
- 梳理现有单 skill 与 skill set 的映射关系，明确哪些是用户可直接安装的官方入口，哪些是内部子 skill。
- 缩小默认暴露面，让新用户优先看到少量官方推荐入口，而不是全部底层 skill。

### Phase 2：重构 README 与发现路径

- README 改为“先路径、后清单”的结构：
  - 我是新手，先装什么
  - 我要做推理，装什么
  - 我要做训练，装什么
  - 我要做 profiling，装什么
- 增加决策树、bundle 对照表、按场景组织的 skill 地图。
- 明确区分：官方主推 skill / bundle、进阶 leaf skill、外部同步 skill。

### Phase 3：建立治理与演进规则

- 建立统一 taxonomy 与命名规范。
- 定义 bundle、router、leaf、external 四类 skill 的角色边界。
- 为核心入口 bundle 与主推 skill 建立更高质量门槛。
- 引入最基本的 usage / feedback 采集方式，为后续整合与去重提供依据。

### Phase 4：建设独立发现入口

- 基于 GitHub Pages 或静态站点提供图形化选择入口。
- 支持按任务、按角色、按阶段筛选推荐安装方案。
- 让 README 成为入口摘要，而把完整导航体验迁移到站点中承载。

总体上，建议优先解决“**用户先装什么、怎么看懂仓库、怎么避免被过多 skills 淹没**”这三个问题，再做长期治理与平台化建设。
