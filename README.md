# bayesian-computational-modeling.-


---

# Bayesian Models of Power, History.

This project explores how **modern data science and machine learning** can be used to explain historical events in a way that is both **quantitative** and **interpretable**. The core idea is that many aspects of “reality” – especially power, conflict, and strategic decision-making – can be modeled with the same tools we use in AI and ML today.

Instead of treating history as a collection of anecdotes, we treat it as a **system of interacting forces** that can be approximated mathematically and simulated computationally.

A public explanation based on this work has accumulated **350,000+ views** on open platforms(like Quora) and has been **shared and endorsed by PhD-level academics**, as well as FAANG companies employers and Top CS Schools graduates(Standford...). which I see as early external validation that this approach is both rigorous and useful.

---

## Conceptual Motivation

In my head, the conceptual starting point was inspired by **Laplace’s demon**:
if one could, in principle, know all the relevant laws and interaction rules, and observe the key variables that govern how humans and states behave, then large parts of the future would become **predictable in a probabilistic sense**.

History is, in effect, an enormous, high-dimensional dataset:
economic indicators, territorial changes, alliances, wars, technological shocks, political decisions, and more. Most of these signals are noisy, heterogeneous, and difficult to connect in a clean way. Seeing the **hidden structure** inside this mess usually requires enormous effort and careful modeling.

This project is an attempt to build such structure explicitly. It is technically demanding and requires **non-trivial engineering** to keep models objective and interpretable, but it also demonstrates how this kind of work could be a **real use-case for AI in the future of the humanities and social sciences**: not just explaining the past, but also giving disciplined insight into what kinds of futures are structurally likely.

---

## Method: A Neuro-Symbolic Framework for Historical Outcomes

In machine-learning terms, the project is a **neuro-symbolic framework** that combines:

* **Causal inference** – to reason about how changing one factor (e.g., colonial resources, alliances, or political structure) would have changed outcomes.
* **Game theory** – to model strategic bargaining between actors (states, coalitions, colonial powers) when they divide territory or make decisions under conflict.
* **Bayesian networks and probabilistic simulation** – to encode uncertainty and propagate it through the system so that we get *distributions* over outcomes, not just single numbers.

The goal is not to produce a mystical “one true number,” but to show that **power, influence, and outcomes can be modeled in a disciplined, mathematical way**, and that those models can yield insights that align with real history—and sometimes highlight **where tension was building before events actually exploded.**

This repository currently contains **two main code paths and explanatory layers**:

1. A **more complex model of how power works**, centered on the **partition of Africa**.
2. A **lighter, more accessible model** applying similar principles to **Rome vs. Carthage and Hannibal**.

Both share the same underlying philosophy:
historical outcomes can be framed as the result of structured constraints, incentives, and resources, not just “great men” or isolated events.

---

## Part I – Power and the Partition of Africa

The first, more complex part of the project studies **how power operates in practice**, using the nineteenth-century **partition of Africa** as a case study.

Historically, European powers drew African borders in a way that was:

* Driven almost entirely by **strategic bargaining** between imperial states.
* Largely indifferent to **indigenous populations**, local cultures, or ethnic boundaries.
* Executed through **unilateral decisions**, often made in European capitals, not in Africa itself.

This episode is a near-perfect laboratory for understanding **how political deliberation and consensus emerge under pure power incentives**, with morality and local self-determination almost entirely ignored. It shows what politics looks like **when only strategy and resources matter**.

In this project, I:

* Reconstructed each colonial power’s **economic model** from that era (trade capacity, industrial strength, existing territories, military capability, etc.).
* Incorporated a range of **relevant variables** (resources, access routes, strategic chokepoints, rivalries, alliances).
* Applied **statistical and computational methods** to simulate how rational, bargaining actors would divide the continent, given those constraints.

The surprising result:

* For almost every major power, the model’s predicted territorial outcome matches actual history within about **3-4% error**.
* The **only major exception is Germany**.

According to the model, **Germany should have received almost twice the colonial territory it actually held**.

At first, this looked like a failure of the model. But on reflection, it pointed toward an important insight:

* Germany **under-achieved** its “expected share” of colonial territory.
* This shortfall plausibly generated **strong internal pressure** to catch up.
* That pressure contributed to the **aggressive expansionism** and insecurity that helped drive Europe toward **World War I**.

In other words, Germany’s path to WWI is not just political narrative; it is a **quantitative anomaly** in the resource/power landscape. A model like this suggests that the structural tension **could have been detected mathematically**, long before open conflict began.

This is more than a curiosity about the past.
Today, we see a **similar pattern** of strategic competition in places like the **Arctic**, where global powers are quietly negotiating over resources, routes, and influence.

The same framework can, in principle, be adapted to:

* Estimate **how influence and resources are likely to be divided** in the Arctic.
* Explore possible future partitions in **modern Africa** or other regions.
* Study **how power “wants” to flow**, given the constraints of geography, economics, and strategy.

The broader claim is not that the future is fully predetermined, but that **political power is, to a surprising degree, mathematically “countable”** when you focus on structure, not personalities.

---

## Part II – A Lighter Model: Rome, Hannibal, and the Importance of Resources

The second part of the project is a **lighter, more accessible model** that still inherits the same methodology and spirit of the first part.

Here, the focus is on a classic question:

> Why did **Rome** ultimately triumph over **Hannibal** and **Carthage**,
> even though Hannibal is often regarded as one of the greatest military commanders in history?

The model encodes:

* Carthage and Rome’s **resource base** (population, economic capacity, naval power, manpower, political stability, strategic position).
* Commander-level attributes (Hannibal, Scipio, Napoleon in comparison), such as:

  * strategic brilliance
  * tactical genius
  * logistics
  * inspiration
  * adaptability
  * political support
  * resource management
* A **Monte Carlo battle simulation** that combines:

  * relative power indices
  * commander effectiveness
  * randomness and uncertainty through probabilistic sampling and logistic outcome functions.

By running these simulations, the code shows that:

* Hannibal’s **individual skill** is extraordinary—his effectiveness score is extremely high.
* However, **resources and structural support** dominate in the long run.

  * Rome has deeper manpower reserves, more political stability, and better sustained mobilization.
  * Carthage fails to provide Hannibal with the **systematic backing** that his campaigns would have required to translate tactical brilliance into permanent victory.

In other words, the model reinforces a hard lesson:

> **Skill alone is not enough.**
> Class, resources, and structural power still matter more than individual excellence.

This is precisely the kind of pattern the framework is designed to highlight:
we can use the **same language of variables, weights, and probabilities** to talk about both:

* the **partition of Africa**, and
* the **Roman victory over Hannibal**,

and, in principle, many other historical outcomes as well.

---

## Why This Matters

Together, these two components show how **modern AI / ML, data science, and probabilistic modeling** can:

* Turn vague historical narratives into **explicit models**.
* Reveal **hidden tensions** and **structural imbalances** that precede major events.
* Provide a testbed for **“what if?” counterfactuals** in a disciplined, quantitative way.
* Suggest how we might one day **forecast certain classes of geopolitical outcomes**, not as prophecy, but as **probabilistic, model-based scenarios**.

The work is technically challenging and demands careful engineering to avoid overclaiming or introducing bias. But it also shows something important about the **future of AI in human sciences**:

> With the right models, we can not only explain parts of the past,
> but also **illuminate which futures are structurally plausible**, and why.

---


# Power Analysis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

First project (Africa.py) is A sophisticated computational framework for analyzing historical geopolitical power dynamics through the lens of machine learning, causal inference, and game theory. This system models colonial-era power allocation among European nations (circa 1890) and provides tools for counterfactual analysis, uncertainty quantification, and modern scenario projection.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Methodology](#methodology)
  - [Power Index Calculation](#power-index-calculation)
  - [Causal Inference Framework](#causal-inference-framework)
  - [Game-Theoretic Allocation](#game-theoretic-allocation)
  - [Uncertainty Quantification](#uncertainty-quantification)
  - [RAG-Enhanced Explanation Generation](#rag-enhanced-explanation-generation)
- [Technical Design Decisions](#technical-design-decisions)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Results & Outputs](#results--outputs)
- [References](#references)

---

## Overview

This framework addresses a fundamental question in quantitative historical analysis: **How can we computationally model the factors that determined colonial power allocation, and what insights can this provide about geopolitical tensions leading to major conflicts?**

The system specifically analyzes the discrepancy between Germany's projected colonial share (based on economic and military indicators) versus its actual historical allocation, quantifying this "colonial frustration" as a contributing factor to WWI tensions.

### Core Hypothesis

Germany's economic and industrial capabilities in the late 19th century warranted a significantly larger colonial empire than it actually possessed. This discrepancy, quantifiable through multi-factor power indices, contributed to geopolitical tensions and aggressive foreign policy.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Factor Power Indexing** | Weighted combination of 8 historical indicators with ML-optimized coefficients |
| **Shapley Value Allocation** | Game-theoretic fair division based on marginal contributions |
| **Structural Causal Modeling** | DAG-based causal inference with do-calculus and counterfactual reasoning |
| **Bayesian Neural Networks** | Uncertainty-aware predictions with variational inference |
| **Attention Mechanisms** | Dynamic factor weighting using transformer-style attention |
| **Monte Carlo Simulation** | Robustness analysis with 1000+ simulation runs |
| **RAG-Enhanced Explanations** | Retrieval-augmented generation for contextual explanations |
| **Modern Scenario Transfer** | Meta-learning for applying historical patterns to contemporary geopolitics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AdvancedGeopoliticalAnalyzer                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │  Query Analysis  │───▶│  Base Predictions │───▶│   Attention    │ │
│  │  (NLP/BERT)      │    │  (Power Index)    │    │   Mechanism    │ │
│  └──────────────────┘    └──────────────────┘    └────────────────┘ │
│           │                       │                       │          │
│           ▼                       ▼                       ▼          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Context Retrieval│    │ Causal Reasoning │    │  Uncertainty   │ │
│  │ (RAG Pipeline)   │    │ (SCM + Do-calc)  │    │ Quantification │ │
│  └──────────────────┘    └──────────────────┘    └────────────────┘ │
│           │                       │                       │          │
│           └───────────────────────┴───────────────────────┘          │
│                                   │                                  │
│                                   ▼                                  │
│                      ┌──────────────────────┐                        │
│                      │ Explanation Generator │                       │
│                      │ (SHAP + Templates)    │                       │
│                      └──────────────────────┘                        │
│                                   │                                  │
│                                   ▼                                  │
│                      ┌──────────────────────┐                        │
│                      │   Formatted Report   │                        │
│                      │   + Visualizations   │                        │
│                      └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### Power Index Calculation

The system computes a composite power index for each nation using a weighted combination of historical indicators:

```
P(country) = Σ wᵢ · fᵢ(xᵢ) · A(country)
```

Where:
- `wᵢ` = learned weight for factor i (via Random Forest feature importance)
- `fᵢ` = transformation function (sigmoid, log-normal, or logarithmic)
- `xᵢ` = raw indicator value
- `A(country)` = colonial ambition index (composite of 5 sub-factors)

**Indicators Used:**

| Indicator | Transformation | Rationale |
|-----------|---------------|-----------|
| Coal Production | `log(x)` | Diminishing returns at scale |
| Naval Tonnage | `(x/1000)^0.7` | Sublinear power projection |
| Population | `(x/1e6)^0.5` | Square root for mobilization capacity |
| GDP | `log(x)` | Economic complexity effects |
| Industrial Capacity | `sigmoid(x/50)` | Saturation at high industrialization |
| Colonial Infrastructure | `sigmoid(x/50)` | Administrative capacity bounds |
| Technology Score | Linear | Composite of 6 tech domains |
| Economic Complexity | Linear | ECI from trade data |

**Why These Transformations?**

- **Logarithmic transforms** capture diminishing marginal returns inherent in economic factors
- **Sigmoid functions** model capacity constraints and saturation effects
- **Power transforms** (e.g., `x^0.7`) balance between linear and logarithmic scaling

### Causal Inference Framework

The `StructuralCausalModel` class implements Pearl's causal hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Causal DAG Structure                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   population ──────────┬──────────────────▶ gdp                 │
│        │               │                     │                   │
│        │               │                     ▼                   │
│        └───────────────┼─────────────────▶ power_index          │
│                        │                     ▲                   │
│   coal_production ─────┼──▶ industrial ──────┤                   │
│        │               │    capacity         │                   │
│        └───────────────┼─────────────────────┤                   │
│                        │                     │                   │
│   tech_level ──────────┼──▶ naval_tonnage ───┘                   │
│        │               │                                         │
│        └───────────────┴──▶ gdp                                  │
│                                                                  │
│   colonial_infrastructure ─────────────────▶ power_index        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Causal Operations Supported:**

1. **Backdoor Adjustment**: Identify confounders and adjust for them
2. **Do-Calculus**: Compute interventional distributions P(Y|do(X))
3. **Counterfactual Analysis**: "What if Germany had Britain's naval tonnage?"

**Why Structural Causal Models over Correlation-Based Approaches?**

- Correlation ≠ Causation: SCMs distinguish between observational and interventional queries
- Counterfactual reasoning is impossible with purely statistical models
- DAGs encode domain knowledge about causal mechanisms

### Game-Theoretic Allocation

Colonial power allocation is modeled as a cooperative game where nations bargain over territorial division.

**Shapley Value Computation:**

```
φᵢ = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] · [v(S∪{i}) - v(S)]
```

This formula computes each nation's "fair share" based on its marginal contribution to all possible coalitions.

**Why Shapley Values?**

- **Axiomatic Foundation**: Uniquely satisfies efficiency, symmetry, null player, and additivity axioms
- **Historical Relevance**: Colonial allocation was effectively a bargaining problem
- **Marginal Contribution**: Captures both absolute power and relative positioning

### Uncertainty Quantification

The framework employs multiple UQ methods:

| Method | Use Case | Advantages |
|--------|----------|------------|
| **Monte Carlo Simulation** | Parameter uncertainty propagation | Non-parametric, handles complex distributions |
| **Bootstrap** | Model uncertainty | Distribution-free confidence intervals |
| **Bayesian Neural Networks** | Epistemic uncertainty | Separates aleatoric vs epistemic uncertainty |
| **Conformal Prediction** | Prediction intervals | Distribution-free coverage guarantees |

**Monte Carlo Implementation:**

```python
for _ in range(n_runs):
    # Resample from parameter distributions
    perturbed_params = sample_from_distributions(historical_db)
    
    # Recompute power indices
    shares = calculate_bargaining_shares(perturbed_params)
    
    # Store for statistical analysis
    results.append(shares)

# Compute statistics
mean, std, percentiles = compute_statistics(results)
```

**Why Multiple UQ Methods?**

- Different sources of uncertainty require different treatments
- Monte Carlo captures parametric uncertainty
- BNNs capture model uncertainty
- Conformal prediction provides guaranteed coverage

### RAG-Enhanced Explanation Generation

The `AdvancedExplanationGenerator` implements a full retrieval-augmented generation pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query ──▶ [Query Understanding] ──▶ [Intent Classification]    │
│                                           │                      │
│                                           ▼                      │
│            ┌──────────────────────────────────────────┐         │
│            │         Parallel Retrieval               │         │
│            ├──────────────────────────────────────────┤         │
│            │  • Web Search (Google Custom Search API) │         │
│            │  • Semantic Search (TF-IDF embeddings)   │         │
│            │  • Knowledge Graph Traversal             │         │
│            │  • Document Corpus Retrieval             │         │
│            └──────────────────────────────────────────┘         │
│                                           │                      │
│                                           ▼                      │
│            [Content Extraction & Analysis]                       │
│            • NER (dbmdz/bert-large-cased-finetuned-conll03)     │
│            • Sentiment (distilbert-finetuned-sst-2)             │
│            • Summarization (facebook/bart-large-cnn)            │
│                                           │                      │
│                                           ▼                      │
│            [Ranking & Fusion] ──▶ [Template-Based Generation]   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why RAG over Pure Generation?**

- **Grounding**: Prevents hallucination by anchoring to retrieved facts
- **Traceability**: Explanations can be traced to sources
- **Updatability**: New information incorporated without retraining

---

## Technical Design Decisions

### Why PyTorch over TensorFlow?

- **Dynamic Computation Graphs**: Essential for varying coalition sizes in Shapley computation
- **Research-Oriented**: Better suited for novel architectures (attention mechanisms, BNNs)
- **Pythonic API**: Cleaner integration with scientific computing stack

### Why NetworkX for Causal Graphs?

- **Mature Graph Algorithms**: Built-in path finding, topological sorting
- **Flexibility**: Easy modification of graph structure
- **Interoperability**: Seamless conversion to/from adjacency matrices for ML

### Why Random Forest for Weight Calibration?

- **Interpretability**: Feature importances directly usable as weights
- **Robustness**: Handles multicollinearity in historical indicators
- **Non-linearity**: Captures complex interactions without explicit specification

### Why Not Deep Learning End-to-End?

- **Sample Size**: Only 7 nations (colonial powers) — insufficient for deep learning
- **Interpretability**: Historical analysis requires explainable models
- **Domain Knowledge**: Structured approach allows incorporating historiographical insights

### Why Multi-Head Attention?

- **Dynamic Weighting**: Factor importance varies by context/country
- **Parallelizable**: Efficient computation across all country pairs
- **Proven Architecture**: Transformer attention is well-understood and debuggable

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Dependencies

```bash
pip install torch numpy pandas scikit-learn networkx scipy matplotlib seaborn tqdm
pip install transformers beautifulsoup4 requests
pip install pyro-ppl shap  # Optional: for advanced Bayesian inference and SHAP
```

### Full Installation

```bash
git clone https://github.com/yourusername/geopolitical-power-analysis.git
cd geopolitical-power-analysis
pip install -r requirements.txt
```

---

## Usage

### Basic Analysis

```python
from main import AdvancedGeopoliticalAnalyzer, AdvancedExplanationGenerator

# Initialize analyzer
analyzer = AdvancedGeopoliticalAnalyzer(
    explanation_generator=AdvancedExplanationGenerator()
)

# Run analysis
query = "Analyze Germany's colonial discrepancy and its impact on WW1 risk"
response = analyzer.analyze_geopolitical_scenario(query)

# Format and display results
print(analyzer.format_response(response))
```

### Modern Scenario Application

```python
# Define modern scenario data
arctic_data = {
    "USA": {
        "population": 331e6,
        "gdp": 21430,
        "tech_level": 0.95,
        "military_expenditure": 778,
        "investment": 0.85,
        "infrastructure": 0.8,
        "diplomatic_influence": 0.9,
    },
    "Russia": {
        "population": 146e6,
        "gdp": 1483,
        "tech_level": 0.75,
        # ... additional parameters
    },
    # ... additional countries
}

# Apply historical model to modern scenario
analysis = analyzer.apply_to_modern_scenario(arctic_data, "Arctic Resource Competition")
```

### Counterfactual Analysis

```python
# What if Germany had Britain's naval tonnage?
counterfactual = analyzer.base_model.causal_model.counterfactual_analysis(
    country="Germany",
    intervention="naval_tonnage",
    value=980  # Britain's naval tonnage
)

print(f"Power index change: {counterfactual['effect']:.2f}")
```

---

## Model Components

### Core Classes

| Class | Purpose |
|-------|---------|
| `EnhancedHyperPowerIndexGenerator` | Main model for power index calculation |
| `StructuralCausalModel` | DAG-based causal inference |
| `MultiHeadAttention` | Dynamic factor weighting |
| `BayesianNeuralNetwork` | Uncertainty-aware neural network |
| `AdvancedUncertaintyQuantification` | Multiple UQ methods |
| `QueryUnderstanding` | NLP-based query analysis |
| `AdvancedExplanationGenerator` | RAG-enhanced explanation generation |
| `AdvancedGeopoliticalAnalyzer` | Orchestration and response synthesis |

### Data Sources

The historical database (`_load_world_bank_1890`) contains:

- **Population**: Census data from ~1890
- **Coal Production**: Million metric tons
- **Naval Tonnage**: Warship displacement (thousands of tons)
- **GDP**: Estimated in contemporary units
- **Industrial Capacity**: Index normalized to Britain = 100
- **Colonial Infrastructure**: Administrative capacity index

---

## Results & Outputs

### Sample Output: Germany's Colonial Discrepancy

```
┌─────────────────────────────────────────────────────────────────┐
│              Colonial Allocation Predictions                     │
├─────────────────────────────────────────────────────────────────┤
│ Country  │ Projected │ Historical │ Discrepancy                 │
├─────────────────────────────────────────────────────────────────┤
│ Britain  │   30.2%   │   32.4%    │   -6.8%                     │
│ France   │   24.1%   │   27.9%    │  -13.6%                     │
│ Germany  │   19.8%   │    8.7%    │ +127.6%  ← Key Finding      │
│ Belgium  │    8.2%   │    7.8%    │   +5.1%                     │
│ Portugal │    6.1%   │    9.5%    │  -35.8%                     │
│ Italy    │    7.3%   │    5.2%    │  +40.4%                     │
│ Spain    │    4.3%   │    3.5%    │  +22.9%                     │
└─────────────────────────────────────────────────────────────────┘

WWI Risk Assessment:
• Tension Factor: 2.34
• Probability of Major Conflict: 73.2% (95% CI: [68.1%, 78.3%])
• Naval Arms Race Correlation: 0.79
• Predicted Diplomatic Incidents/Year: 4.2
```

### Visualization Suite

The framework generates:

1. **Discrepancy Bar Chart**: Projected vs. historical colonial shares
2. **Uncertainty Plot**: Error bars showing 90% confidence intervals
3. **PCA Projection**: 2D visualization of national power factors
4. **t-SNE Embedding**: Non-linear clustering of similar nations
5. **Sensitivity Heatmap**: Parameter sensitivity analysis
6. **Causal Effect Plot**: Horizontal bar chart of causal effects

---

## References

### Methodological Foundations

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Shapley, L. S. (1953). "A Value for n-Person Games". *Contributions to the Theory of Games*.
3. Vaswani, A., et al. (2017). "Attention Is All You Need". *NeurIPS*.
4. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions". *NeurIPS*.

### Historical Sources

1. Kennedy, P. (1987). *The Rise and Fall of the Great Powers*. Random House.
2. Maddison, A. (2007). *Contours of the World Economy, 1-2030 AD*. Oxford University Press.
3. Förster, S. (1999). "Dreams and Nightmares: German Military Leadership and the Images of Future Warfare".

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---


Second project of same research (Lite_Carthage.py) uses a probabilistic model, ML-based feature weighting, Monte Carlo simulation, and NLP explanation tools to analyze why Carthage lost to Rome in the Second Punic War. Project computes power indices, win probabilities for key battles, and scenario/counterfactual outcomes, then generates a structured human-readable narrative. Lite_Carthage.py provides a simpler, dependency-light variant focused on the core quantitative model, compared to previus project.
Here’s a clean README you can drop straight into `README.md` for the Carthage project (han.py). You can adjust the title / tone later if you want it more “researchy”.

---

# Carthage vs Rome – Probabilistic War & Commander Simulator

This repository contains a **toy but serious** computational model of the Second Punic War, focused on the question:

> **“Why did Carthage lose to Rome, despite Hannibal’s genius?”**

The core script, `han.py`, builds a **probabilistic world-model** of Carthage and Rome, simulates key battles, and then generates a **human-readable narrative** explaining the result, together with plots.

It is not meant as a precise historical model, but as an example of how **modern data science & ML-style techniques** can represent and analyze a complex historical situation.

---

## Key Features

* **Structured world model**

  * Encodes population, economic resources, naval power, manpower, political stability, and strategic position.
  * Includes detailed commander profiles (Hannibal, Scipio Africanus, Napoleon) with traits like strategic brilliance, resource management, and political support.

* **Machine-learning–inspired resource weights**

  * Uses a `RandomForestRegressor` to derive feature importances and convert them into **resource weights** for each factor.

* **Battle simulator with uncertainty**

  * Computes **win probabilities** for battles (Cannae, Zama) via Monte Carlo simulation and a logistic win function.

* **Scenario & counterfactual analysis**

  * Compares resource-heavy vs commander-heavy scenarios.
  * Adds a “What if Carthage fully supported Hannibal?” counterfactual and estimates its impact on power and win chances.

* **Narrative explanation generator**

  * `AdvancedExplanationGenerator` turns raw numbers into a **structured explanation**, with sections on:

    * Power comparison
    * Commander analysis
    * Key battles
    * Resource vs commander importance
    * Comparison with Napoleon
    * Counterfactual analysis

* **Visualizations**

  * Saves plots for:

    * Overall power index: `carthage_rome_power_comparison.png`
    * Carthage win probability under different scenarios: `carthage_win_probability.png`
    * Commander effectiveness comparison: `commander_effectiveness.png`

---

## Technologies Used (and Why)

The code intentionally uses a mix of modern DS/ML tools to show how they can model a historical “world”:

* **NumPy / SciPy** – numerical operations, random sampling, logistic functions.
* **scikit-learn**

  * `RandomForestRegressor` – used to estimate **relative importance** of different resources and convert that into weights for the power index.
  * `TfidfVectorizer` – builds document embeddings for a small pseudo-RAG corpus of historical snippets.
* **PyTorch**

  * Defines a `MultiHeadAttention` module for future experiments with learned factor weighting (not strictly required in the default pipeline, but included as an advanced extension hook).
* **Transformers (Hugging Face)**

  * Tries to load:

    * `facebook/bart-large-cnn` for summarization
    * `distilbert-base-uncased-finetuned-sst-2-english` for sentiment
  * If model loading fails, the code falls back to a simpler text pipeline.
  * Currently, the explanation text is largely template-based, but the NLP hooks are ready for future enhancement.
* **Matplotlib / Seaborn** – visualize power indices and win probabilities.
* **pandas** – convenient handling of small structured numerical results.
* **(Optional) Pyro / SHAP**

  * Imported as optional dependencies for future Bayesian inference and feature-importance explanations. The current version does not yet use them in the main pipeline.

---

## How the Model Works

### 1. Historical Data & World State

`PunicWarsAnalyzer` starts by loading a synthetic “historical database”:

```python
self.historical_db = {
    "population": {...},
    "economic_resources": {...},
    "naval_power": {...},
    "manpower": {...},
    "political_stability": {...},
    "strategic_position": {...},
}
```

Each entry is a mean + std for Carthage, Rome, and their allies.
These are **stylized values**, not precise historical measurements, but chosen to reflect the usual narrative:

* Carthage: stronger navy, trade economy, weaker political cohesion
* Rome: stronger manpower, better political stability, robust military system

### 2. Learning Resource Weights (Random Forest)

The method `_calculate_resource_factors` uses a `RandomForestRegressor` to learn which resources matter most:

1. Constructs feature vectors:

   * `[population, economic_resources, naval_power, manpower, political_stability, strategic_position]`
2. Defines target `y`:

   * `1.0` for Rome (winner)
   * `0.0` for Carthage (loser), plus small random noise
3. Fits `RandomForestRegressor` and reads `feature_importances_`.

Those feature importances are turned into **continuous weight functions**:

```python
"Population": lambda f: importance * log(population),
"Economic":  lambda f: importance * log(economic_resources),
"Naval":     lambda f: importance * (naval_power / 1000)**0.7,
...
```

This creates a **data-driven “power index”** that reflects how much each resource contributes to overall strength.

### 3. Power Index

`calculate_power_index(faction)` sums all weighted factors:

```python
index = (
    weights["Population"](faction)
    + weights["Economic"](faction)
    + weights["Naval"](faction)
    + weights["Manpower"](faction)
    + weights["Political"](faction)
    + weights["Strategic"](faction)
)
```

It returns:

```python
{"mean": index, "std": index * 0.1}
```

The `std` is set to 10% of the mean to model uncertainty.

In a typical run, you’ll see something like:

* Carthage power index ≈ 5.47
* Rome power index ≈ 5.15

So the model actually says: **Carthage’s raw resource/position index is slightly higher**, but that alone doesn’t decide the war.

### 4. Commander Model

`_init_commander_profiles` defines trait vectors for:

* Hannibal
* Scipio Africanus
* Napoleon

Each has attributes `strategic_brilliance`, `tactical_genius`, `logistical_skill`,
`inspiration_ability`, `adaptability`, `political_support`, `resource_management` (all on a 0–10 scale).

`analyze_commander_impact(name)` computes an overall **effectiveness score** as a weighted sum of traits, giving more weight to strategic and tactical ability, but also including political support & resource management. It also identifies strengths (traits ≥ 9) and weaknesses (traits ≤ 7).

This is how the model encodes the common story:

* Hannibal – genius on the battlefield, weaker political backing & resources.
* Napoleon – similar genius, but with much stronger state control.

### 5. Battle Simulation (Cannae & Zama)

`simulate_battle_outcome(attacker, defender, attacker_commander, defender_commander)`:

1. Computes the power indices for attacker and defender.
2. Computes commander effectiveness for both.
3. Derives:

   * `power_ratio = attacker_power_mean / defender_power_mean`
   * `commander_ratio = attacker_effectiveness / defender_effectiveness`
4. Runs a **Monte Carlo simulation** (1000 iterations):

   * Samples attacker/defender power from normal distributions around their means.
   * Calculates a **logistic win probability**:
     [
     \text{win_prob} = \sigma\big(\log(\text{power_ratio_sample}) + 0.3 \cdot \text{commander_ratio}\big)
     ]
   * Draws Bernoulli trials to estimate `attacker_win_probability`.

The result matches intuition:

* At **Cannae** (Hannibal vs Varro, Carthage attacking), Carthage has a relatively high win probability.
* At **Zama** (Scipio vs Hannibal, Rome attacking near Carthage), Rome’s probability is higher.

### 6. Resource vs Commander Scenarios

`analyze_resource_vs_commander_importance()` defines three scenarios:

* Balanced (resources and commanders weighted equally)
* Resource Heavy
* Commander Heavy

It then estimates Carthage’s win probability in each case, based on:

* Scaled power indices (resource advantage)
* Scaled commander impact (commander advantage)

This lets you see things like:

* In a **resource-heavy** world, Carthage might still lose if Rome dominates structurally.
* In a **commander-heavy** world, Hannibal’s genius becomes much more decisive.

### 7. Napoleon Comparison

`compare_hannibal_napoleon()` interprets commander effectiveness combined with:

* Political support
* Resource management

to compute support/freedom scores and highlight the key contrast:

> Napoleon had complete control of France’s resources; Hannibal did not have equivalent control over Carthage.

This feeds directly into the narrative part of the explanation.

### 8. Counterfactual Carthage

The code defines a hypothetical “Carthage with full support”:

```python
carthage_with_support = {
    "Military Leadership": 9.5,
    "Economic Resources": 8.5,
    "Political Structure": 8.0,
    "Alliances": 7.0,
    "Strategic Position": 7.0,
    "Naval Power": 9.0,
    "Manpower": 7.5,
}
```

`_calculate_power_index_from_factors` turns this 0–10-factor dict into a **normalized synthetic power index** using a fixed set of weights.

Then the code plugs this into another logistic expression to produce an “outcome probability” if Carthage had fully supported Hannibal. This yields text like:

> “With full political support and better resource management, Carthage’s power index would have been: 0.82, giving about a 13.7% chance of defeating Rome.”

The numbers are not meant as literal truth; they illustrate how a change in structural variables affects outcomes in the model.

### 9. Explanation Generator (NLP / RAG Hook)

`AdvancedExplanationGenerator` does three things:

1. Builds a tiny **corpus** of historical mini-documents (Hannibal campaign, Zama, Carthaginian politics, Roman military system, Napoleon’s resources).
2. Creates TF-IDF embeddings for this corpus (preparing for retrieval-augmented workflows).
3. Generates a structured explanation from `analysis_results` with sections:

   * Power comparison
   * Commander analysis
   * Key battles
   * Resource vs commander importance
   * Comparison with Napoleon
   * Counterfactual analysis

It currently uses **template-based text**, but optionally loads Hugging Face summarization / sentiment models for richer future behavior.

So the full pipeline is:

> numerical world model → simulations → structured narrative explanation

---

## How to Run

### Requirements

You’ll need a Python environment (3.10+ recommended) with:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `torch`
* `transformers` (optional but recommended)
* `tqdm`
* `seaborn`
* `requests`, `beautifulsoup4` (imported for future extensions)

Install the core packages with:

```bash
pip install numpy pandas matplotlib scikit-learn torch transformers tqdm seaborn requests beautifulsoup4
```

### Running the Script

From the repository directory:

```bash
python han.py
```

This will:

1. Print a **multi-section explanation** of why Carthage lost.
2. Save three PNG files into the current directory:

   * `carthage_rome_power_comparison.png`
   * `carthage_win_probability.png`
   * `commander_effectiveness.png`

If Hugging Face models can’t be downloaded, the script will print a warning and fall back to basic text processing, but the main analysis will still work.

---

## Interpreting the Output

A typical run might say:

* Carthage’s **power index** is slightly higher than Rome’s (e.g. 5.47 vs 5.15).
* Hannibal’s **commander effectiveness** is very high, but his weaknesses are political support and resource management.
* **Cannae** shows a high Carthaginian win probability;
  **Zama** shows Rome favored.
* In **resource-heavy** scenarios, resources/political structure dominate.
  In **commander-heavy** scenarios, Hannibal can “carry” more.
* The **counterfactual** shows that with full support, Carthage’s chances improve, but structural disadvantages still matter.

The conclusion emphasizes:

> Carthage did not lose because Hannibal was weak, but because its **political and resource system** failed to fully support him, whereas Rome could mobilize its entire state.

---



---


