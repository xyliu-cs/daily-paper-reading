# 每日论文报告 — Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

## 结构化摘要

| 维度 | 内容 |
|---|---|
| **标题** | Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses |
| **机构** | 复旦大学 (Fudan University)、北京大学 (Peking University)、Shanghai Qiji Zhifeng Co., Ltd |
| **作者** | Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui |
| **时间** | 2026-04-30 (arXiv v3) |
| **发表** | arXiv preprint (arXiv:2604.25850v3) |
| **链接** | https://github.com/china-qijizhifeng/agentic-harness-engineering |
| **总结** | 研究旨在解决编码智能体（coding agent）的"外壳"（harness，即系统提示、工具、中间件、技能、记忆等模型外可编辑组件）只能依靠人工不断手工调整、难以伴随基模演进而自动迭代的问题。论文提出 **AHE (Agentic Harness Engineering)**：以"组件可观测、经验可观测、决策可观测"三层观测能力驱动的闭环——把每次编辑变成一份带预测、可被下一轮事实证伪的合约，由 Evolve Agent 在文件级粒度上对 harness 做出可回滚的修改。在 Terminal-Bench 2 上仅迭代 10 次即把 pass@1 由 69.7% 提升到 77.0%，超过人工设计的 Codex-CLI（71.9%）以及自演化基线 ACE 与 TF-GRPO，并能跨任务（SWE-bench-verified）和跨模型族（GPT/Gemini/DeepSeek/Qwen）零再训练迁移，带来 +5.1～+10.1 pp 的稳定增益。|

---

## 1. 研究背景和问题

编码智能体在 GitHub issue 解决与多步终端任务等长程软件工程基准上进步显著，但性能不仅取决于底层 LLM，同样取决于围绕模型的可编辑外壳——系统提示、工具集、中间件（middleware）、技能（skills）、记忆（memory）。这一组件集合统称为 *harness*。当前 harness 工程依赖人工逐版本观察轨迹、手写编辑，难以追上模型迭代速度，使得"模型潜力"和"harness 能兑现的潜力"之间出现日益拉大的差距。

### 1.1 核心假设

论文提出的核心假设是：harness 自动演化的瓶颈不在 agent 能力，而在 *可观测性 (observability)*——只要给演化 agent 提供清晰的动作空间、结构化的经验语料、以及每次编辑的可证伪契约，它就能稳定收敛到更好的 harness 设计，无需依赖盲目试错。论文的核心研究问题是："How can an evolution agent **jointly and stably** evolve all editable components of a coding agent's harness?"（原文 §1）。

---

## 2. 方法

![图1：AHE 整体闭环结构。组件、轨迹经验、编辑决策三个可观测面被串成一个闭环，每次编辑都成为下一轮可被证伪的预测。](AHE_fig/AHE-Figure2-1.png)

AHE 把 harness 优化定义为一个由另一个 agent 驱动的封闭回路，并通过三层可观测面将其结构化。**组件可观测 (Component Observability)** 借助 NexAU 框架将 harness 解耦为七类正交组件文件（系统提示、工具描述、工具实现、中间件、技能、子 agent 配置、长程记忆），使每条失败模式都能映射到唯一组件类，且每次编辑落到一次 git commit 上，天然带来 file-level diff 与回滚粒度。**经验可观测 (Experience Observability)** 采用 Agent Debugger 把每条任务轨迹落成可被 shell/脚本工具导航的文件树，由调试 agent 产出 *per-task analysis*，再聚合成 *benchmark-level overview*——把数百万 token 的原始 rollout 蒸馏为约 10K token 的分层证据语料，以"渐进披露 (progressive disclosure)"的方式喂给上游。**决策可观测 (Decision Observability)** 要求 Evolve Agent 在每次编辑时同时给出一份 change manifest：失败证据、推断根因、目标修复、可预期修复集与潜在回归集；下一轮以任务级 delta 与之求交，自动判定该编辑是 confirm 还是 rollback，把"自我合理化"替换为"可衡量合约"。Algorithm 1 把这三层组装成一次外循环：rollout → clean → 上一轮 manifest 归因与回滚 → 分层蒸馏 → 工作区编辑 + 新 manifest → git commit；演化只能写 harness 工作区文件，runs/tracer/verifier/LLM 配置以及种子系统提示被声明为只读或不可删除，从机制上禁止"作弊式自我修改"。

---

## 3. 实验

### 3.1 实验设置

| 维度 | 名称 | 数量 |
|---|---|---|
| **模型** | GPT-5.4 (high / medium / xhigh，作为三角色共享基模)；跨模型迁移基线：qwen-3.6-plus、gemini-3.1-flash-lite-preview、deepseek-v4-flash | 主基模 1 个 + 跨族评测 5 个配置 |
| **训练（演化）** | 在 Terminal-Bench 2 上做 10 次 AHE 迭代，每任务每轮 k=2 次 rollout，整轮约 32 小时 | 89 任务（4 easy / 55 medium / 30 hard），每轮 178 次 rollout |
| **评测** | Terminal-Bench 2（演化目标）；SWE-bench-verified（跨基准迁移） | 89 任务 + 500 任务（覆盖 7 个仓库：django、sympy、sphinx-doc、matplotlib、scikit-learn、pydata、astropy）|
| **指标** | pass@1（k 次 rollout 的二元成功率均值）、tokens/trial（每次 trial 平均 prompt+completion token，单位千） | 2 |

对照组包括三个人工设计 harness（opencode、terminus-2、Codex-CLI）和两个自演化基线（ACE、TF-GRPO），后两者与 AHE 共享 NexAU₀ 种子，仅演化层不同。基础设施使用 Harbor 调度器 + 全新 E2E 远程沙箱 + InMemoryTracer + Langfuse；Agent Debugger 并发度 16，单任务超时 600s。

### 3.2 实验解析

#### 3.2.1 主结果：AHE 在 Terminal-Bench 2 上的迭代演化曲线

![图2：AHE 在 Terminal-Bench 2 上把 bash-only 种子推过所有人工和自演化基线，三角色共享同一基模。](AHE_fig/AHE-Figure1-1.png)

- **图表内容**：横轴为自动演化迭代序号 (1–10)，纵轴为 Terminal-Bench 2 上的 pass@1 (%)。实线为 AHE best-so-far，虚线为单轮 pass@1，三条水平参考线分别是 TF-GRPO (72.3)、Codex (71.9) 与 ACE (68.9)。曲线上的标注框给出了关键编辑（如 contract-first workflow + 可调 shell 超时、publish-state guard、cross-step risk monitor、post-success hard-block）。
- **揭示关系**：单轮 pass@1 是非单调的（出现回退），但 best-so-far 在 10 轮内累计单调上升，提示 AHE 的"manifest + 回滚"机制能在不放弃试错的情况下抑制回归。
- **关键数据**：从 NexAU₀ 种子的 69.7% → 77.0%（+7.3 pp）；超过 Codex-CLI (71.9%) 5.1 pp，超过 TF-GRPO 4.7 pp，超过 ACE 8.1 pp；按难度切分 AHE 在 Easy 100%、Medium 88.2%、Hard 53.3%，仅 Hard 一档略低于人工 Codex（56.7%）。

#### 3.2.2 跨模型族迁移：冻结 harness 直接换基模

![图3：在 GPT-5.4 high 上演化得到的 AHE workspace，被原样套到其它五个基模上重新评测，与同基模 NexAU₀ 种子对比。](AHE_fig/AHE-Figure3-1.png)

- **图表内容**：每对柱状图代表一个基模，左浅蓝为 NexAU₀ 种子 pass@1，右深蓝为同 harness 套上 AHE 演化结果后的 pass@1，柱顶标注绝对增益。
- **揭示关系**：所有 5 个跨模型配置都呈正向迁移（+2.3 pp 到 +10.1 pp），且越远离 GPT-5.4 high 演化点的基模，绝对收益越大——deepseek-v4-flash +10.1 pp，qwen-3.6-plus +6.3 pp，gemini-3.1-flash-lite +5.1 pp，都高于 GPT-5.4 medium/xhigh 上的 +2.3 pp。这暗示 AHE 演化进 tools/middleware/memory 的协调模式是"通用工程经验"而非对 GPT-5.4 高推理档的过拟合，越欠饱和的基模越能从这些外部脚手架中获益。

#### 3.2.3 组件级消融：增益究竟落在哪里

![图4：将单一 AHE 组件（memory / tool / middleware / system_prompt）替换回 NexAU₀，其余三项保持种子默认。](AHE_fig/AHE-Table3-1.png)

- **图表内容**：表格按 All / Easy / Medium / Hard 四档列出 NexAU₀ 种子、四种 "+ X only" 单组件变体，以及完整 AHE。
- **揭示关系**：四个单组件中三个（memory、tool、middleware）单独换入即可正向超过种子（+5.6 / +3.3 / +2.2 pp），唯独 "+system_prompt only" 反而 -2.3 pp。组件的失败面互不重叠却又非加性叠加：单 memory 在 Hard 上 63.3% 反超完整 AHE 的 53.3%，说明 memory、middleware、system prompt 都在推同一类闭环验证型修复，彼此竞争长程预算，AHE 当前在 medium-heavy 演化目标下牺牲了部分 Hard memory 收益。这一现象揭示 harness 演化存在"组件交互非加性"的结构性瓶颈，是论文明确指出的未来工作方向。

#### 3.2.4 自归因精度：演化 agent 是否真的"知道自己在修什么"

![图5：9 轮演化中，Evolve Agent 自宣称的"修复"与"潜在回归"在下一轮真实 task delta 上的精度/召回，与随机基线对比。](AHE_fig/AHE-Figure4-1.png)

- **图表内容**：左侧两组柱代表 Fix precision / Fix recall，右侧两组代表 Regression precision / Regression recall；深色为 cross-iteration 平均，浅色为 random-prediction baseline。
- **揭示关系**：在"修复"侧，agent 命中率显著高于随机（precision 33.7% vs 6.5%、recall 51.4% vs 10.6%，约 5×），说明每次编辑确实瞄准了一个真实可被证伪的目标；但在"回归"侧只有 11.8% / 11.1%，仅约 2× 随机基线，意味着 agent 虽然能解释"为什么这条改动应该有用"，却基本无法预知"它会让哪些任务变坏"——这正是单轮 pass@1 出现非单调跳变的根源。论文将"回归预见"列为后续自演化循环最明确的提升方向。

---

## 4. 局限性与未来工作

### 4.1 原文描述

论文专设 *Limitations* 章节，明确给出三点局限。**Benchmark scope**：演化只在 Terminal-Bench 2 上驱动、迁移仅在 SWE-bench-verified 上验证，对更广的编程语言、仓库级部署、人在回路工作流尚未测试。**Evolution operating point**：AHE 的步预算与单任务超时是为 GPT-5.4 high 拟合的，跨模型迁移的数字混合了 harness 可移植性与"操作点耦合"，同一族内跨推理档 +2.3/+7.3/+2.3 pp 的非单调性即源于此，需在多操作点重跑回路才能解耦。**Self-modification governance**：AHE 把编辑限定在 harness 工作区、靠 manifest 归因与文件级回滚做最小护栏，但缺少完整的 guardrail 栈，长程清理与误用防护仍不完善，作者明确把 AHE 定位为"受控研究原型"而非成熟的自我改进系统。此外，§4.4.2 与 §4.4.1 把"组件交互非加性导致增益封顶"和"agent 对回归的自归因近似随机"列为最清晰的后续工作。

### 4.2 模型总结

AHE 把"observability"作为 agent 自演化的可执行接口，是一种很有工程美感的范式——把每次编辑都变成可证伪合约的设计，规避了大多数自演化系统"奖励信号噪声远大于信号"的核心困境，与 Algorithm 1 的"先归因再蒸馏再编辑"次序相互成全。但研究的高方差性也很明显：演化目标只有 89 个任务、单轮约 178 次 rollout 难以稳定揭示长尾失败模式；四组件中 memory 与 middleware 大量重叠在"closure-style verification"上，提示真实的 harness 设计空间维度可能比七元分解更低；自归因在回归侧近乎随机，意味着当循环规模继续扩大时，最先失效的不是"能不能找到好编辑"，而是"能不能预知坏编辑"。可能的延伸方向包括：(1) 把 change manifest 改造成显式回归测试集而非自然语言断言，让 verifier 直接消费；(2) 在多操作点（推理档、超时预算）上联合演化以解开模型/harness 耦合；(3) 引入 component-interaction-aware 的目标，避免 medium-heavy 优化吞掉 Hard 收益；(4) 用 AHE 输出的 manifest ledger 反过来训练 harness-aware 奖励模型，闭合"observability-driven evolution"与"model-side training"两条独立的能力轴。（由 Claude Opus 4.7 生成）
