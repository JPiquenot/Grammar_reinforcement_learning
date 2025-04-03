# Grammar Reinforcement Learning (GRL)

**Official Implementation** of the ICLR 2025 paper:  
**"Grammar Reinforcement Learning: Path and Cycle Counting in Graphs with a Context-Free Grammar and Transformer Approach"**  
Authors: Jason Piquenot, Maxime Bérar, Romain Raveaux, Pierre Héroux, Jean-Yves Ramel, Sébastien Adam

## Overview

This repository contains the source code for Grammar Reinforcement Learning (GRL), a novel reinforcement learning algorithm that integrates Monte Carlo Tree Search (MCTS) with a transformer architecture modeling a Pushdown Automaton (PDA) within a Context-Free Grammar (CFG) framework. GRL is designed to discover efficient matrix-based formulas for counting paths and cycles in graphs, a fundamental problem in various domains such as network analysis, computer science, biology, and social sciences.

Key contributions of GRL include:

1. **CFG-Based Transformer Framework**: A novel approach for generating transformers that operate within a CFG, enabling the modeling of complex grammatical structures.
2. **Reinforcement Learning Integration**: The development of GRL for optimizing formulas within grammatical structures using reinforcement learning techniques.
3. **Efficient Graph Substructure Counting**: Discovery of novel formulas for graph substructure counting, leading to significant computational improvements—enhancing efficiency by factors of two to six compared to state-of-the-art methods.

For more details, refer to the [paper](https://arxiv.org/abs/2410.01661).

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch 
- Torch Geometric
- tqdm

Install the required packages using pip:

```bash
pip install torch torch-geometric tqdm
```

### Running the GRL Agent

To launch the GRL agent using MCTS for grammar-based search:

```bash
python acting.py <agent_id:int> <ntask:int> <without_policy:True|False>
```

- `agent_id`: Identifier for the agent, used for saving outputs.
- `ntask`: Task identifier (e.g., graph prediction variant).
- `without_policy`: Boolean flag to disable loading a pretrained grammar model.

**Example**:

```bash
python acting.py 0 1 False
```

### Training the Grammar Model (Pre-learning)

To pre-train the grammar model using collected agent rollouts:

```bash
python learning.py
```

This script collects rollouts from agents, constructs a dataset, and pre-trains the grammar model using reinforcement learning techniques.

## Core Concepts

### Grammar Definition

The grammar rules for symbolic reasoning are defined as:

```
E -> (E*M) | (N@E) | (E@N) | A | J
N -> (N*M) | (N*N) | I
M -> (M@M) | (E@E)
```

Each production has an associated cost guiding exploration. The `libs.grammartools` module builds the grammar structure.



### Actor & MCTS

- `acting.py` launches agents using MCTS to explore grammar-based programs.
- Agent episodes are generated and scored using graph datasets.
- Best programs are saved for training.

### Training

- Collected program sequences are saved to disk.
- `learning.py` uses these rollouts to optimize the grammar transformer via reinforcement learning.

## Dataset

The code uses a dataset of subgraphs located in `dataset/subgraphcount/`. Ensure this directory exists and is populated before running training.

## Outputs

- Saved agent programs: `save/agent*/save_agent*.dat`
- Result logs: `save/agent*/results_agent*.dat`
- Trained grammar policy: `save/grammartest/grammar.dat`

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{piquenot2025grammar,
  title={Grammar Reinforcement Learning: Path and Cycle Counting in Graphs with a Context-Free Grammar and Transformer Approach},
  author={Piquenot, Jason and Bérar, Maxime and Raveaux, Romain and Héroux, Pierre and Ramel, Jean-Yves and Adam, Sébastien},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

