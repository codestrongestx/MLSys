# Track A Solver — Target Spec

## Goal

Build a solver that produces **minimum-latency schedules** for all 25 benchmarks,
verified by a **spec-complete evaluator** that matches Google's hidden scorer.

---

## 1. Evaluator (tools/evaluate_solution.py)

The evaluator is the foundation — every solver improvement is meaningless if
we can't verify it locally. It must handle every legal solution shape.

### 1.1 Current state

- Handles: Pointwise-only subgraphs, MatMul + optional Pointwise tail
- Verified against 11/12 PROBLEM.md examples
- Gap: **chained MatMul subgraphs** (Example 5: split-K pipelining) — rejects
  two MatMuls in one subgraph

### 1.2 Target state

Must correctly evaluate **any valid subgraph**, including:

- [x] Single Pointwise op
- [x] Chain of Pointwise ops (fused)
- [x] Single MatMul op (with split-K)
- [x] MatMul + Pointwise tail (fused)
- [ ] **Chained MatMuls** (e.g. `(A @ B) @ C` with split-K, Example 5)
- [ ] **MatMul with Pointwise inputs** (e.g. bias add before matmul)
- [ ] **Arbitrary DAG subgraphs** — the spec allows any connected set of ops
      in a subgraph as long as it's topologically valid. The evaluator should
      not assume chain structure.
- [ ] **Recomputation** — the spec allows an op to appear in multiple
      subgraphs (Example 3B: Op0 appears in both subgraph 0 and 1). The
      evaluator must allow this.
- [ ] **Padding cost** — when spatial granularity `(w, h)` is smaller than
      `native_granularity`, compute cost should reflect padding waste. Current
      code may not handle this correctly in all paths.

### 1.3 Validation checks the evaluator must enforce

- Every op covered at least once across all subgraphs
- Working set fits in `fast_memory_capacity` at every tile step
- `tensors_to_retain` only lists output tensors of that subgraph
- Final subgraph retains nothing (all graph outputs end in slow memory)
- Subgraph-internal op order is topologically valid
- Traversal order is a valid permutation of tile indices
- Reported `subgraph_latencies` match computed values (within tolerance)

### 1.4 Testing

- All 12 PROBLEM.md example strategies must pass with exact expected values
- Targeted edge-case tests: zero-cost ops, single-element tensors, tensors
  with non-power-of-2 dimensions, maximum memory pressure scenarios
- Roundtrip: solver output → evaluator → no errors, for all 5 benchmarks

---

## 2. Solver (source/mlsys.cpp)

### 2.1 Current state

- Greedy topological scheduler
- Chain-only subgraph grouping (linear: A→B→C, must be single-output→single-consumer Pointwise)
- Granularity search over power-of-2 candidates for `[w, h, k]`
- Traversal: raster vs snake (pick better)
- Tensor retention: 1-step lookahead (retain output if it helps next subgraph)
- Score heuristic: `latency / num_ops` (prefers large, cheap subgraphs)

### 2.2 Target capabilities

#### A. Subgraph formation

The solver should consider subgraph structures beyond linear chains:

1. **Diamond / skip-connection grouping** — when an intermediate tensor feeds
   two downstream ops, consider grouping them together (Example 3C pattern:
   `[Op1, Op2]` where both consume retained Tensor1).

2. **Recomputation** — when retaining an intermediate is too expensive
   (memory), consider recomputing it in a later subgraph. The spec explicitly
   allows ops to appear in multiple subgraphs (Example 3B: Op0 recomputed).

3. **Chained MatMul fusion** — group consecutive MatMuls with split-K to keep
   intermediates ephemeral (Example 5B: `[Op0(MatMul), Op1(MatMul)]` with
   `k=32`).

4. **Multi-output subgraphs** — currently only handles single-output chains.
   Some graphs may benefit from subgraphs that produce multiple outputs.

#### B. Scheduling order

1. **Search over topological orderings** — the current greedy picks one root
   at a time. Try multiple orderings (beam search, or enumerate promising
   alternatives) and keep the best.

2. **Global retention planning** — instead of 1-step lookahead, plan retention
   across the full schedule. This is essentially a form of register allocation:
   fast memory slots are "registers" and tensors are "values." Use a global
   cost model to decide what to retain.

3. **Iterative refinement** — after producing an initial greedy solution,
   try local perturbations:
   - Swap adjacent subgraph order
   - Merge two consecutive subgraphs
   - Split a large subgraph into two
   - Toggle retention on/off for a tensor
   - Try different granularities for a subgraph
   Accept changes that reduce total latency (hill climbing / simulated
   annealing).

#### C. Granularity optimization

1. **Finer candidate generation** — current candidates are powers of 2 down
   to 16. Consider also factors of the tensor dimension, and odd sizes that
   evenly divide the tensor (avoiding partial last tiles).

2. **Per-tile latency awareness** — last tiles may be smaller (partial).
   The current code already handles this but verify it accounts for padding.

3. **Joint w/h/k search** — currently searches w×h×k independently. Some
   combinations may have non-obvious interactions (e.g. smaller k frees
   memory for larger w×h).

#### D. Traversal order

1. **Beyond raster/snake** — try Z-order (Morton), Hilbert curve, or
   problem-specific orderings that maximize data reuse for the specific
   tensor shapes involved.

2. **Retention-aware traversal** — when a tensor is retained from a previous
   subgraph, the traversal order should account for what's already in fast
   memory.

### 2.3 Architecture

The solver should be structured as:

```
main()
  → ReadProblem()
  → InitialSolve()        // greedy baseline (current approach, improved)
  → Refine()              // iterative improvement loop (time-budgeted)
  → ValidateSolution()
  → WriteSolution()
```

The `Refine()` loop runs until the timeout approaches (leave ~1s margin),
continuously trying perturbations and accepting improvements. This is where
most of the scoring gains will come from on larger benchmarks.

### 2.4 Performance targets

The solver must complete within the per-benchmark timeouts:

| Nodes | Timeout |
|-------|---------|
| ~5    | 2 sec   |
| ~19   | 5 sec   |
| ~32   | 15 sec  |
| ~63   | 30 sec  |
| ~103  | 60 sec  |
| TBD   | 120 sec |

For small benchmarks (≤19 nodes), near-optimal solutions via exhaustive or
branch-and-bound search may be feasible. For large benchmarks, the greedy +
refinement approach is necessary.

---

## 3. Build & test infrastructure

### 3.1 Current state

- `source/Makefile` — simple single-file build
- No automated test suite
- Manual: run solver, then run evaluator, compare

### 3.2 Target state

- [ ] **`make test`** — runs solver on all 5 benchmarks, pipes each through
      evaluator, fails on any error
- [ ] **`make test-examples`** — runs evaluator on all 12 PROBLEM.md example
      strategies, verifies expected latencies
- [ ] **`make benchmark`** — runs solver on all 5 benchmarks, reports total
      latency and per-benchmark breakdown, compares against previous best
      (stored in a baseline file)
- [ ] **Static linking** — `make release` produces a statically linked binary
      for Ubuntu 22.04 LTS (the submission target)

---

## 4. Milestones

### M1: Complete evaluator
- Fix chained MatMul evaluation (Example 5)
- Add recomputation support (ops in multiple subgraphs)
- Add arbitrary DAG subgraph evaluation
- All 12 PROBLEM.md examples pass
- Add `make test-examples`

### M2: Improved subgraph formation
- Diamond / skip-connection grouping
- Recomputation strategy
- Chained MatMul fusion with split-K
- Add `make test` and `make benchmark`

### M3: Global optimization
- Multi-step retention planning
- Search over topological orderings
- Iterative refinement loop (hill climbing)
- Time-budgeted execution

### M4: Polish & submission
- Advanced traversal orders
- Finer granularity candidates
- Static linking for submission
- Profile on large benchmarks
- Final latency comparison and writeup
