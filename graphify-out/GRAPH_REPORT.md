# Graph Report - D:\DL  (2026-09-13)

## Corpus Check
- Corpus is ~34,586 words - fits in a single context window. You may not need a graph.

## Summary
- 17 nodes · 17 edges · 5 communities
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.85)
- Token cost: 25,285 input · 2,419 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Cycle-aware RUL Modeling|Cycle-aware RUL Modeling]]
- [[_COMMUNITY_SEI Physics Constraints|SEI Physics Constraints]]
- [[_COMMUNITY_NASA SOH Representation|NASA SOH Representation]]
- [[_COMMUNITY_Adaptive Twin Deployment|Adaptive Twin Deployment]]
- [[_COMMUNITY_Proposal and Evidence|Proposal and Evidence]]

## God Nodes (most connected - your core abstractions)
1. `Bat-T-GNN` - 5 edges
2. `EM-PINN Framework` - 4 edges
3. `PINNE (Physics-Informed Neural Network-based Estimation)` - 4 edges
4. `Adaptive Physics-Informed Battery Digital Twin` - 4 edges
5. `Adaptive Physics-Informed Battery Digital Twin` - 3 edges
6. `SEI Layer Growth Model` - 2 edges
7. `State of Health (SOH) Estimation` - 2 edges
8. `NASA PCoE Dataset` - 1 edges
9. `PDE Loss` - 1 edges
10. `Replay-Buffer Learner` - 1 edges

## Surprising Connections (you probably didn't know these)
- `Bat-T-GNN` --conceptually_related_to--> `PINNE (Physics-Informed Neural Network-based Estimation)`  [INFERRED]
  3.pdf → 4.pdf
- `Adaptive Physics-Informed Battery Digital Twin` --references--> `Bat-T-GNN`  [EXTRACTED]
  phse.pdf → 3.pdf
- `Adaptive Physics-Informed Battery Digital Twin` --references--> `PINNE (Physics-Informed Neural Network-based Estimation)`  [EXTRACTED]
  phse.pdf → 4.pdf

## Import Cycles
- None detected.

## Communities (5 total, 0 thin omitted)

### Community 0 - "Cycle-aware RUL Modeling"
Cohesion: 0.50
Nodes (4): Bat-T-GNN, Cycle-Aware Patching, PINN-RUL (Physics-Informed Consistency Loss), T-PATCHGNN

### Community 1 - "SEI Physics Constraints"
Cohesion: 0.67
Nodes (4): EM-PINN Framework, PDE Loss, SEI Layer Growth Model, State of Health (SOH) Estimation

### Community 2 - "NASA SOH Representation"
Cohesion: 0.67
Nodes (3): Momentum Contrastive Learning, Physics-Guided Data Augmentation (PDA), PINNE (Physics-Informed Neural Network-based Estimation)

### Community 3 - "Adaptive Twin Deployment"
Cohesion: 0.67
Nodes (3): Adaptive Physics-Informed Battery Digital Twin, NASA PCoE Dataset, Replay-Buffer Learner

### Community 4 - "Proposal and Evidence"
Cohesion: 0.67
Nodes (3): Adaptive Physics-Informed Battery Digital Twin, NASA Lithium-Ion Battery Aging Dataset, Yang et al. (2025) - Applied Energy

## Knowledge Gaps
- **9 isolated node(s):** `NASA PCoE Dataset`, `Replay-Buffer Learner`, `Cycle-Aware Patching`, `PINN-RUL (Physics-Informed Consistency Loss)`, `T-PATCHGNN` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Bat-T-GNN` connect `Cycle-aware RUL Modeling` to `NASA SOH Representation`, `Proposal and Evidence`?**
  _High betweenness centrality (0.175) - this node is a cross-community bridge._
- **Why does `PINNE (Physics-Informed Neural Network-based Estimation)` connect `NASA SOH Representation` to `Cycle-aware RUL Modeling`, `Proposal and Evidence`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Why does `Adaptive Physics-Informed Battery Digital Twin` connect `Proposal and Evidence` to `Cycle-aware RUL Modeling`, `NASA SOH Representation`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **What connects `NASA PCoE Dataset`, `PDE Loss`, `Replay-Buffer Learner` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._