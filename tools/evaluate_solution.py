#!/usr/bin/env python3

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tensor:
    width: int
    height: int


@dataclass(frozen=True)
class Op:
    op_type: str
    inputs: list[int]
    outputs: list[int]
    base_cost: int


@dataclass(frozen=True)
class Problem:
    tensors: list[Tensor]
    ops: list[Op]
    fast_memory_capacity: int
    slow_memory_bandwidth: float
    native_width: int
    native_height: int
    producer: list[int]
    consumers: list[list[int]]
    graph_outputs: set[int]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def tensor_area(tensor: Tensor) -> int:
    return tensor.width * tensor.height


def bytes_to_time(num_bytes: int, bandwidth: float) -> float:
    return num_bytes / bandwidth


def load_problem(path: Path) -> Problem:
    raw = json.loads(path.read_text())
    tensors = [Tensor(w, h) for w, h in zip(raw["widths"], raw["heights"])]
    ops = [
        Op(op_type, inputs, outputs, base_cost)
        for op_type, inputs, outputs, base_cost in zip(
            raw["op_types"], raw["inputs"], raw["outputs"], raw["base_costs"]
        )
    ]
    producer = [-1] * len(tensors)
    consumers = [[] for _ in tensors]
    for op_index, op in enumerate(ops):
        for tensor_id in op.outputs:
            if producer[tensor_id] != -1:
                raise ValueError(f"tensor {tensor_id} has multiple producers")
            producer[tensor_id] = op_index
        for tensor_id in op.inputs:
            consumers[tensor_id].append(op_index)
    graph_outputs = {idx for idx, users in enumerate(consumers) if not users}
    return Problem(
        tensors=tensors,
        ops=ops,
        fast_memory_capacity=int(raw["fast_memory_capacity"]),
        slow_memory_bandwidth=float(raw["slow_memory_bandwidth"]),
        native_width=int(raw["native_granularity"][0]),
        native_height=int(raw["native_granularity"][1]),
        producer=producer,
        consumers=consumers,
        graph_outputs=graph_outputs,
    )


def validate_lengths(solution: dict) -> int:
    keys = [
        "subgraphs",
        "granularities",
        "tensors_to_retain",
        "traversal_orders",
        "subgraph_latencies",
    ]
    lengths = [len(solution[key]) for key in keys]
    if len(set(lengths)) != 1:
        raise ValueError(f"solution arrays have mismatched lengths: {dict(zip(keys, lengths))}")
    return lengths[0]


def validate_subgraph_order(problem: Problem, ops: list[int]) -> None:
    seen = set()
    for op_index in ops:
        if op_index in seen:
            raise ValueError(f"subgraph repeats op {op_index}")
        seen.add(op_index)
        for tensor_id in problem.ops[op_index].inputs:
            producer = problem.producer[tensor_id]
            if producer != -1 and producer in seen:
                continue
            if producer != -1 and producer in ops:
                raise ValueError(f"subgraph order is not topological at op {op_index}")


def external_inputs(problem: Problem, ops: list[int]) -> list[int]:
    inside = set(ops)
    seen = []
    seen_set = set()
    for op_index in ops:
        for tensor_id in problem.ops[op_index].inputs:
            producer = problem.producer[tensor_id]
            if producer == -1 or producer not in inside:
                if tensor_id not in seen_set:
                    seen_set.add(tensor_id)
                    seen.append(tensor_id)
    return seen


def boundary_outputs(problem: Problem, ops: list[int]) -> list[int]:
    inside = set(ops)
    seen = []
    for op_index in ops:
        for tensor_id in problem.ops[op_index].outputs:
            if tensor_id in seen:
                continue
            if tensor_id in problem.graph_outputs:
                seen.append(tensor_id)
                continue
            if any(consumer not in inside for consumer in problem.consumers[tensor_id]):
                seen.append(tensor_id)
    return seen


def validate_retain_list(problem: Problem, ops: list[int], retain: list[int]) -> None:
    produced = set(boundary_outputs(problem, ops))
    for tensor_id in retain:
        if tensor_id not in produced:
            raise ValueError(f"tensor {tensor_id} cannot be retained by subgraph {ops}")


def traversal_pairs(rows: int, cols: int, traversal: list[int] | None) -> list[tuple[int, int]]:
    total = rows * cols
    if traversal is None:
        return [(idx // cols, idx % cols) for idx in range(total)]
    if len(traversal) != total:
        raise ValueError("traversal length does not match tile count")
    if sorted(traversal) != list(range(total)):
        raise ValueError("traversal order is not a permutation")
    return [(idx // cols, idx % cols) for idx in traversal]


def validate_shapes(problem: Problem, ops: list[int], granularity: list[int]) -> None:
    w, h, _ = granularity
    if w <= 0 or h <= 0:
        raise ValueError("non-positive granularity")
    for op_index in ops:
        for tensor_id in problem.ops[op_index].inputs + problem.ops[op_index].outputs:
            tensor = problem.tensors[tensor_id]
            if w > tensor.width or h > tensor.height:
                raise ValueError(f"granularity exceeds tensor shape for tensor {tensor_id}")


def eval_pointwise(
    problem: Problem,
    ops: list[int],
    granularity: list[int],
    retain: set[int],
    prev_retained: set[int],
) -> float:
    output_tensor = problem.tensors[problem.ops[ops[-1]].outputs[0]]
    w, h, _ = granularity
    ext_inputs = external_inputs(problem, ops)
    retained_input_bytes = sum(
        tensor_area(problem.tensors[tensor_id])
        for tensor_id in ext_inputs
        if tensor_id in prev_retained
    )
    final_output = problem.ops[ops[-1]].outputs[0]
    retain_output = final_output in retain
    compute_cost = sum(problem.ops[op_index].base_cost for op_index in ops)
    total = 0.0
    for row in range(ceil_div(output_tensor.height, h)):
        tile_h = min(h, output_tensor.height - row * h)
        for col in range(ceil_div(output_tensor.width, w)):
            tile_w = min(w, output_tensor.width - col * w)
            load_bytes = sum(
                tile_h * tile_w
                for tensor_id in ext_inputs
                if tensor_id not in prev_retained
            )
            output_bytes = tile_h * tile_w
            reserved_output = tensor_area(output_tensor) if retain_output else output_bytes
            working_set = retained_input_bytes + load_bytes + reserved_output
            if working_set > problem.fast_memory_capacity:
                raise ValueError("OOM in pointwise subgraph")
            store_bytes = 0 if retain_output else output_bytes
            total += max(
                float(compute_cost),
                bytes_to_time(load_bytes + store_bytes, problem.slow_memory_bandwidth),
            )
    return total


def eval_matmul_chain(
    problem: Problem,
    ops: list[int],
    granularity: list[int],
    traversal: list[int] | None,
    retain: set[int],
    prev_retained: set[int],
) -> float:
    root = problem.ops[ops[0]]
    if root.op_type != "MatMul":
        raise ValueError("expected matmul root")
    if any(problem.ops[op_index].op_type != "Pointwise" for op_index in ops[1:]):
        raise ValueError("evaluator only supports matmul with optional pointwise tail")

    output_tensor = problem.tensors[problem.ops[ops[-1]].outputs[0]]
    lhs = problem.tensors[root.inputs[0]]
    rhs = problem.tensors[root.inputs[1]]
    w, h, k = granularity
    k_dim = lhs.width
    ext_inputs = external_inputs(problem, ops)
    extra_pointwise = [
        tensor_id
        for tensor_id in ext_inputs
        if tensor_id not in (root.inputs[0], root.inputs[1])
    ]
    final_output = problem.ops[ops[-1]].outputs[0]
    retain_output = final_output in retain
    pointwise_compute = sum(problem.ops[op_index].base_cost for op_index in ops[1:])

    lhs_retained = root.inputs[0] in prev_retained
    rhs_retained = root.inputs[1] in prev_retained
    retained_input_bytes = 0
    if lhs_retained:
        retained_input_bytes += tensor_area(lhs)
    if rhs_retained:
        retained_input_bytes += tensor_area(rhs)
    retained_input_bytes += sum(
        tensor_area(problem.tensors[tensor_id])
        for tensor_id in extra_pointwise
        if tensor_id in prev_retained
    )

    rows = ceil_div(output_tensor.height, h)
    cols = ceil_div(output_tensor.width, w)

    if k == k_dim:
        order = traversal_pairs(rows, cols, traversal)
        cached_row = -1
        cached_col = -1
        compute = root.base_cost * k_dim / problem.native_width + pointwise_compute
        total = 0.0
        for row, col in order:
            tile_h = min(h, output_tensor.height - row * h)
            tile_w = min(w, output_tensor.width - col * w)
            row_bytes = 0 if lhs_retained else tile_h * k_dim
            col_bytes = 0 if rhs_retained else k_dim * tile_w
            pointwise_bytes = sum(
                tile_h * tile_w
                for tensor_id in extra_pointwise
                if tensor_id not in prev_retained
            )
            output_bytes = tile_h * tile_w
            reserved_output = tensor_area(output_tensor) if retain_output else output_bytes
            working_set = retained_input_bytes + row_bytes + col_bytes + pointwise_bytes + reserved_output
            if working_set > problem.fast_memory_capacity:
                raise ValueError("OOM in matmul subgraph")
            load_bytes = pointwise_bytes
            if not lhs_retained and cached_row != row:
                load_bytes += tile_h * k_dim
            if not rhs_retained and cached_col != col:
                load_bytes += k_dim * tile_w
            store_bytes = 0 if retain_output else output_bytes
            total += max(
                float(compute),
                bytes_to_time(load_bytes + store_bytes, problem.slow_memory_bandwidth),
            )
            cached_row = row
            cached_col = col
        return total

    if ops[1:]:
        raise ValueError("split-k with pointwise tail is unsupported")

    total = 0.0
    for row in range(rows):
        tile_h = min(h, output_tensor.height - row * h)
        for col in range(cols):
            tile_w = min(w, output_tensor.width - col * w)
            red_steps = ceil_div(k_dim, k)
            for step in range(red_steps):
                step_k = min(k, k_dim - step * k)
                lhs_bytes = 0 if lhs_retained else tile_h * step_k
                rhs_bytes = 0 if rhs_retained else step_k * tile_w
                output_bytes = tile_h * tile_w
                reserved_output = tensor_area(output_tensor) if retain_output else output_bytes
                working_set = retained_input_bytes + lhs_bytes + rhs_bytes + reserved_output
                if working_set > problem.fast_memory_capacity:
                    raise ValueError("OOM in split-k matmul")
                store_bytes = 0
                if step + 1 == red_steps and not retain_output:
                    store_bytes = output_bytes
                compute = root.base_cost * step_k / problem.native_width
                total += max(
                    float(compute),
                    bytes_to_time(lhs_bytes + rhs_bytes + store_bytes, problem.slow_memory_bandwidth),
                )
    return total


def evaluate_subgraph(
    problem: Problem,
    ops: list[int],
    granularity: list[int],
    retain: list[int],
    traversal: list[int] | None,
    prev_retained: set[int],
) -> float:
    validate_subgraph_order(problem, ops)
    validate_shapes(problem, ops, granularity)
    validate_retain_list(problem, ops, retain)
    if not ops:
        raise ValueError("empty subgraph")
    if all(problem.ops[op_index].op_type == "Pointwise" for op_index in ops):
        return eval_pointwise(problem, ops, granularity, set(retain), prev_retained)
    return eval_matmul_chain(problem, ops, granularity, traversal, set(retain), prev_retained)


def evaluate_solution(problem: Problem, solution: dict) -> tuple[float, list[float]]:
    num_steps = validate_lengths(solution)
    prev_retained: set[int] = set()
    covered = set()
    actual_latencies: list[float] = []

    for step in range(num_steps):
        ops = solution["subgraphs"][step]
        granularity = solution["granularities"][step]
        retain = solution["tensors_to_retain"][step]
        traversal = solution["traversal_orders"][step]
        reported = float(solution["subgraph_latencies"][step])
        actual = evaluate_subgraph(problem, ops, granularity, retain, traversal, prev_retained)
        actual_latencies.append(actual)
        if not math.isclose(actual, reported, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(
                f"subgraph {step} latency mismatch: reported={reported:.6f} actual={actual:.6f}"
            )
        covered.update(ops)
        prev_retained = set(retain)

    missing = [idx for idx in range(len(problem.ops)) if idx not in covered]
    if missing:
        raise ValueError(f"solution does not cover every op: missing {missing}")
    if prev_retained:
        raise ValueError("final subgraph retains tensors; graph outputs must be in slow memory at the end")
    return sum(actual_latencies), actual_latencies


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <problem.json> <solution.json>", file=sys.stderr)
        return 1

    problem = load_problem(Path(sys.argv[1]))
    solution = json.loads(Path(sys.argv[2]).read_text())

    try:
        total, latencies = evaluate_solution(problem, solution)
    except ValueError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"OK total_latency={total:.6f}")
    for step, latency in enumerate(latencies):
        print(f"step[{step}] latency={latency:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
