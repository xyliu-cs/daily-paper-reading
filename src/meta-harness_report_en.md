# Daily Paper Reading — Meta-Harness: End-to-End Optimization of Model Harnesses

## Structured Summary

| Dimension | Content |
|---|---|
| **Title** | Meta-Harness: End-to-End Optimization of Model Harnesses |
| **Institution** | Stanford, MIT, KRAFTON |
| **Authors** | Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn |
| **Date** | 2026 (Preprint) |
| **Venue** | Preprint |
| **Link** | Project page: https://yoonholee.com/meta-harness/ ; Code: https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact |
| **Summary** | This work addresses the problem that LLM harnesses—the code determining what information to store, retrieve, and present to a model—are still designed largely by hand. The authors propose Meta-Harness, an outer-loop system that uses a coding agent proposer with filesystem access to all prior candidates' source code, execution traces, and scores to automatically search over harness code. On online text classification, Meta-Harness improves over the state-of-the-art hand-designed harness (ACE) by 7.7 points while using 4× fewer context tokens; on math reasoning it yields a 4.7-point average gain across five held-out models; and on TerminalBench-2 it surpasses the best hand-engineered baselines. |

---

## 1. Background and Problem

The performance of LLM systems depends critically on their harness—the code that determines what to store, retrieve, and show to the model—with harness choice alone producing up to a 6× performance gap on the same benchmark. Despite growing interest in harness engineering, this process remains largely manual. Existing text optimizers are poorly matched to the harness optimization setting because they compress feedback too aggressively: they are memoryless, condition only on scalar scores, or restrict feedback to short templates or LLM-generated summaries, losing the long-range diagnostic information needed to trace downstream failures to earlier harness decisions.

### 1.1 Core Hypothesis

The central hypothesis is that effective harness search requires the proposer to have selective access to the full history of prior experience—source code, execution traces, and scores—via a filesystem, rather than operating from lossy summaries or compressed per-candidate feedback. As validated by the ablation study: "Access to raw execution traces is the key ingredient for enabling harness search" (Table 3).

---

## 2. Method

![Figure 1: Meta-Harness search loop architecture](meta-harness_fig/meta-harness-Figure2-1.png)

Meta-Harness is an outer-loop system that searches over harness code by exposing complete search history through a filesystem to a coding agent proposer. The search loop operates in three stages: (1) the proposer (implemented as Claude Code with Opus-4.6) selectively inspects all prior candidates' source code, execution traces, and evaluation scores via standard terminal tools (grep, cat), then proposes a new harness; (2) the system evaluates the proposed harness on evaluation tasks; (3) all logs—proposed code, reasoning traces, evaluation scores—are stored in a new filesystem directory, and the loop repeats.

The critical distinction from prior text optimizers lies in the scale of available feedback: a single Meta-Harness evaluation can produce up to 10 million tokens of diagnostic information, roughly three orders of magnitude beyond the largest feedback budgets in prior text optimization settings (Table 1). The proposer reads a median of 82 files per iteration in the most demanding setting, referencing over 20 prior candidates per step. Search occurs in code space, allowing the proposer to modify harnesses at the level of algorithmic structure—from retrieval and memory logic to full program rewrites—rather than filling templates or applying predefined mutation operators. The system maintains a Pareto frontier over evaluated harnesses but imposes no parent-selection rule, leaving diagnosis and edit decisions entirely to the proposer.

---

## 3. Experiments

### 3.1 Experimental Setup

| Dimension | Name | Quantity |
|---|---|---|
| **Models** | GPT-OSS-120B (text classification), GPT-OSS-20B/GPT-5.4-nano/GPT-5.4-mini/Gemini-3.1-Flash-Lite/Gemini-3-Flash (math), Claude Opus 4.6/Claude Haiku 4.5 (coding) | 8+ |
| **Training** | Not found (Meta-Harness does not train model weights; it searches over harness code) | Not found |
| **Evaluation** | Text classification: USPTO-50k/Symptom2Disease/LawBench + 9 OOD datasets; Math: 200 IMO-level problems; Coding: TerminalBench-2 (89 tasks) | 3 domains, multiple benchmarks |
| **Metrics** | Accuracy (%), Pass Rate (%), Context tokens (Ctx) | 3 |

### 3.2 Experimental Analysis

### 3.2.1 Online Text Classification

![Figure 2: Search-set accuracy over evaluations for text optimizers](meta-harness_fig/meta-harness-Figure4-1.png)

- **Figure Content**: This figure shows the best-so-far accuracy on the search set as a function of the number of harness evaluations for all compared text optimizers. The x-axis represents harness evaluations and the y-axis represents best search-set accuracy (%).
- **Revealed Insights**: Meta-Harness matches the final accuracy of OpenEvolve and TTT-Discover within just 4 evaluations (versus their ~40), and continues improving to end more than 10 points above all baselines—demonstrating both 10× faster convergence and superior final performance.

On the test set (Table 2), Meta-Harness achieves 48.6% average accuracy, outperforming ACE by 7.7 points and MCE by 8.6 points, while using only 11.4K context tokens (versus 50.8K for ACE and 28.5K for MCE). The ablation study (Table 3) confirms that full filesystem access with execution traces is the critical component: scores-only reaches 34.6% median accuracy versus 50.0% for the full interface.

**Detailed Setup**:  
- Optimization Sets:  
  - LawBench (Law): Predict criminal charges (215 classes) based on case descriptions  
  - Symptom2Disease (S2D): Predict diseases (22 classes) based on symptom descriptions  
  - USPTO-50k: Predict precursor reactants from product molecules (180 classes)  
- Test Sets: Held-out test splits of the same three datasets  
- Initial Harness: zero-shot, few-shot, ACE, and MCE  
- Final Harness:  
![Figure 5: Final Framework for Text Classification](meta-harness_fig/1.png)


### 3.2.2 Retrieval-Augmented Math Reasoning

![Figure 3: Math reasoning results on 200 IMO-level problems](meta-harness_fig/meta-harness-Table6-1.png)

- **Figure Content**: This table reports pass@1 accuracy averaged over three samples per problem on 200 IMO-level math problems across five held-out models, with absolute improvement over the no-retrieval baseline in parentheses.
- **Revealed Insights**: The Meta-Harness-discovered retrieval strategy improves over the no-retrieval baseline across all five held-out models, with a 4.7-point average gain (38.8% vs. 34.1%). Crucially, the harness was optimized on a single model (GPT-OSS-20B) but transfers to four unseen models, demonstrating cross-model generalization.
- **Key Data**: The discovered harness uses a lexical router assigning queries to one of four subject-specific retrieval policies (combinatorics, geometry, number theory, default), all built on BM25 without requiring additional dense encoders.

**Detailed Setup**:  
The model is enhanced by enabling it to retrieve examples from a large corpus, since solutions often share reusable proof patterns. Prior reasoning trajectories therefore contain information that the model can leverage during inference.  
- Optimization Set: 250-problem search set (OlympiadBench + Omni-MATH hard)  
- Test Set: 200 evaluation problems from previously unseen IMO-level problems (IMO-AnswerBench, IMO-ProofBench, and ArXivMath)  
- Initial Harness: zero-shot, few-shot, and ACE  
- Final Harness:  
![Figure 6: Final Framework for Mathematical Reasoning](meta-harness_fig/2.png)


### 3.2.3 Agentic Coding on TerminalBench-2

![Figure 4: TerminalBench-2 leaderboard results](meta-harness_fig/meta-harness-Table7-1.png)

- **Figure Content**: This table shows pass rates on TerminalBench-2 for various harnesses on two base models (Claude Opus 4.6 and Claude Haiku 4.5). The "Auto" column indicates whether the harness was automatically discovered.
- **Revealed Insights**: On Opus 4.6, Meta-Harness achieves 76.4% pass rate, ranking #2 among all agents (only behind ForgeCode at 81.8%, whose results could not be reproduced from public code). On Haiku 4.5, Meta-Harness achieves 37.6%, ranking #1 among all agents. It is the only automatically discovered harness on this competitive leaderboard.
- **Key Data**: The discovered modification is an environment bootstrapping step (~80 lines of code) that gathers a sandbox snapshot before the agent loop begins, eliminating 2–4 wasted exploratory turns on dependency-heavy tasks.

**Detailed Setup**:  
- Optimization Set: TerminalBench-2 evaluates LLM agents on 89 challenging tasks that require long-horizon execution, full autonomy under complex dependencies, and extensive domain knowledge  
- Test Set: TerminalBench-2  
- Initial Harness: Terminus 2, and Terminus-KIRA  
- Final Harness:  
![Figure 7: Final Framework for Agent Coding](meta-harness_fig/3.png)

---

## 4. Limitations and Future Work

### 4.1 Original Description

The authors acknowledge that: (1) their experiments demonstrate harness search with one particular strong coding-agent proposer (Claude Code), and a broader study of how the effect varies across proposer agents remains for future work; (2) a natural next step is to co-evolve the harness and model weights, letting the strategy shape what the model learns and vice versa; (3) the findings reflect a recurring pattern in machine learning (citing Rich Sutton's "The Bitter Lesson"): once a search space becomes accessible, stronger general-purpose agents can outperform hand-engineered solutions.

### 4.2 Model Summary

The primary limitation of Meta-Harness is its dependence on the capability of the proposer coding agent—the method's effectiveness may fluctuate significantly with different underlying agents. The computational cost of search is also substantial (millions of tokens of diagnostic information per iteration), which may limit applicability in resource-constrained settings. The TerminalBench-2 evaluation uses an overlapping search/test split; although the authors mitigate overfitting through manual inspection and regex-based audits, this remains an evaluation concern. Promising future directions include extending Meta-Harness to multi-agent collaboration scenarios, developing more compute-efficient search strategies, and exploring joint optimization of harness code and model weights. (Generated by Claude Opus 4.6 model)
