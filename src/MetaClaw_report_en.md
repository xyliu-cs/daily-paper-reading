# Daily Paper Reading — MetaClaw: An Agent That Meta-Learns and Evolves in the Wild

## Structured Summary

| Dimension | Content |
|---|---|
| **Title** | MetaClaw: Just Talk – An Agent That Meta-Learns and Evolves in the Wild |
| **Institution** | UNC-Chapel Hill, Carnegie Mellon University, UC Santa Cruz |
| **Authors** | Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao |
| **Date** | March 2026 |
| **Venue** | arXiv preprint |
| **Link** | https://github.com/aiming-lab/MetaClaw |
| **Summary** | This work addresses the problem that deployed LLM agents remain static and cannot adapt as user needs evolve. The core method is MetaClaw, a continual meta-learning framework that jointly maintains an evolving skill library and a base LLM policy through two complementary mechanisms: gradient-free skill-driven fast adaptation and gradient-based opportunistic policy optimization triggered during user-inactive windows. Experiments show that skill-driven adaptation improves accuracy by up to 32% relative, the full pipeline advances Kimi-K2.5 from 21.4% to 40.6% (nearly matching GPT-5.2's 41.1% baseline), and skill injection alone improves cross-domain robustness by 18.3%. |

---

## 1. Background and Problem

In the field of LLM agents, deployed systems are typically trained once and served unchanged, creating a fundamental tension: they must serve users continuously while their capabilities grow stale as the task distribution drifts. Existing adaptation approaches fall into three categories—memory-based methods that store raw trajectories without extracting transferable patterns, skill-based methods that maintain static libraries disconnected from weight optimization, and RL-based methods that update weights but ignore data validity after behavioral changes. The core problem this paper addresses is how to unify fast behavioral adaptation with slow policy optimization into a coherent framework that enables agents to evolve continuously in production without service downtime.

### 1.1 Core Hypothesis

MetaClaw is built on the observation that two fundamentally different timescales of adaptation are naturally complementary: behavioral heuristics can be distilled within seconds from a single failure and injected immediately, while cross-task policy improvement requires gradient-based optimization over many trajectories on a timescale of minutes to hours. The two mechanisms are mutually reinforcing—a better policy produces more informative failures for skill synthesis, and richer skills yield higher-reward trajectories for policy optimization.

---

## 2. Method

![Figure 1: Overview of the MetaClaw framework](2603.17187v1_fig/2603.17187v1-Figure1-1.png)

MetaClaw defines the agent's meta-model as M=(θ, S), where θ denotes the base LLM policy parameters and S is an evolving skill library of reusable behavioral instructions. The framework improves this meta-model through two complementary loops operating at different timescales. **Skill-driven fast adaptation** is gradient-free: an LLM skill evolver analyzes failure trajectories and synthesizes new behavioral instructions (e.g., "create .bak before modifying any file", "use ISO 8601 with timezone offset") that are injected via the system prompt and take effect immediately with zero service downtime. **Opportunistic policy optimization** uses RL with a process reward model (PRM) via GRPO and cloud LoRA fine-tuning to update model weights θ, but only during user-inactive windows. The Opportunistic Meta-Learning Scheduler (OMLS) monitors three idle signals: configured sleep windows, system keyboard/mouse inactivity, and Google Calendar event occupancy. To prevent stale reward contamination, the framework introduces a **skill generation versioning** mechanism that strictly separates support data (failure trajectories consumed by skill evolution) from query data (post-adaptation trajectories used for RL updates), ensuring policy optimization always trains on data reflecting the agent's current adapted behavior. The entire system is built on a proxy-based architecture requiring no local GPU to scale to production-size LLMs.

---

## 3. Experiments

### 3.1 Experimental Setup

| Dimension | Name | Quantity |
|---|---|---|
| **Models** | GPT-5.2, Kimi-K2.5 | 2 |
| **Training** | MetaClaw-Bench simulated workday task stream (5-day RL training run) | Not explicitly stated |
| **Evaluation** | MetaClaw-Bench Part I (30 days, 346 Q), Part II (14 days, 588 Q), AutoResearchClaw (23 stages) | 934 + 23 stages |
| **Metrics** | Accuracy (Acc.), File-check Completion Rate (Compl.), Stage Retry Rate, Refine Cycle Count, Pipeline Stage Completion, Composite Robustness Score | 6 |

### 3.2 Experimental Analysis

### 3.2.1 Main Results on MetaClaw-Bench

![Table 1: Main results on MetaClaw-Bench Parts I and II](2603.17187v1_fig/2603.17187v1-Table1-1.png)

- **Figure/Table Content**: Table 1 reports accuracy and file-check completion rates for five model–condition combinations (GPT-5.2 Baseline/Skills, Kimi-K2.5 Baseline/Skills/Full) across both benchmark parts.
- **Revealed Insights**: MetaClaw consistently improves over baselines across both models, adaptation modes, and benchmark parts. Skill injection benefits the weaker model Kimi-K2.5 more (+32.2% relative on Part I), while the full pipeline advances Kimi-K2.5 from 21.4% to 40.6%, nearly closing the gap with GPT-5.2's baseline (41.1%). This suggests MetaClaw is particularly valuable for deploying capable-but-not-SOTA models.
- **Key Data**: MetaClaw (Full) achieves an 8.25× improvement in Kimi-K2.5's end-to-end task completion on Part I (2.0%→16.5%) and a 185% relative gain in file-check completion on Part II (18.2%→51.9%).

### 3.2.2 Per-Day Accuracy Trends

![Figure 2: Per-day accuracy over 30 simulated workdays (3-day rolling average)](2603.17187v1_fig/2603.17187v1-Figure2-1.png)

- **Figure/Table Content**: Figure 2 plots per-day accuracy for all five conditions over 30 days, with solid lines for GPT-5.2 and dashed lines for Kimi-K2.5. The x-axis represents simulated workdays and the y-axis represents accuracy.
- **Revealed Insights**: All conditions show a consistent accuracy decline from early days (>50%) to late days (<30%), confirming the benchmark's increasing difficulty. MetaClaw's advantage is most pronounced in the mid-range (day 11–22), where MetaClaw (Full) reaches peak accuracy near 0.8 around day 19–20. The early days are too simple for adaptation to matter, and the late days are too complex for accumulated knowledge alone.

### 3.2.3 Cross-Domain Generalization on AutoResearchClaw

![Table 2: MetaClaw (Skills-Only) on AutoResearchClaw](2603.17187v1_fig/2603.17187v1-Table2-1.png)

- **Figure/Table Content**: Table 2 reports four robustness metrics for skills-only adaptation (no RL) on AutoResearchClaw, a 23-stage autonomous research pipeline.
- **Revealed Insights**: Skill injection alone, without any gradient-based updates, achieves consistent improvements: stage retry rate decreases by 24.8%, refine cycles drop by 40.0%, pipeline completion improves from 18/19 to 19/19 stages, and composite robustness score rises by 18.3% (0.714→0.845). This demonstrates that MetaClaw's lightweight skill injection mechanism transfers effectively to structurally different, long-horizon agentic workflows beyond CLI tasks.

---

## 4. Limitations and Future Work

### 4.1 Original Description

The paper acknowledges that idle-window detection depends on user configuration (sleep schedule, keyboard idle threshold, Google Calendar integration), which may not generalize to all deployment environments. Additionally, MetaClaw-Bench is an authored simulation rather than a collection of real user sessions, so the absolute magnitude of gains may not transfer directly to production workloads.

### 4.2 Model Summary

MetaClaw represents a principled first attempt at unifying skill evolution with policy optimization for continuously deployed LLM agents, with the skill generation versioning mechanism being a particularly elegant solution to the stale reward contamination problem. However, several limitations warrant further investigation: (1) the skill library grows monotonically without pruning or conflict resolution, which may degrade retrieval quality and inflate prompt length over extended deployments; (2) evaluation is limited to CLI tasks and research pipelines—applicability to multimodal, multi-agent, or safety-critical scenarios remains unexplored; (3) the RL training dynamics over longer deployment horizons (months rather than days) may face stability and catastrophic forgetting challenges. Promising future directions include adaptive idle detection strategies, automatic skill library curation and deduplication, and collaborative skill sharing across multiple users or agent instances. (Generated by Claude Opus 4.6 model)
