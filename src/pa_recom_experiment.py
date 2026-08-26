import csv
import gzip
import json
import math
import os
import pickle
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gerrychain import Graph, Partition
from gerrychain.constraints import contiguous, within_percent_of_ideal_population
from gerrychain.proposals import recom
from gerrychain.updaters import Tally, cut_edges


# ============================================================
# Project paths
# ============================================================
# This script is expected to be inside the project's src/ folder:
#
# project/
# ├── data/
# │   └── PA_VTDs.json
# ├── figures/
# ├── results/
# ├── checkpoints/
# └── src/
#     └── pennsylvania_weighted_recom_10000_checkpoint.py
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "PA_VTDs.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

for directory in (FIGURES_DIR, RESULTS_DIR, CHECKPOINTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================
POPULATION_COLUMN = "TOT_POP"
DISTRICT_COLUMN = "2011_PLA_1"
NUMBER_OF_DISTRICTS = 18
POPULATION_TOLERANCE = 0.02

# TOTAL_TRANSITIONS excludes the initial state. Therefore, a completed run
# contains state 0 plus states 1,...,10000.
TOTAL_TRANSITIONS = 10_000
CHECKPOINT_INTERVAL = 1_000

# Use the same three seeds for every beta, for example:
# RANDOM_SEED = 42, 123, or 1234
RANDOM_SEED = 42


# BETA = 0       -> baseline under the same proposal and constraints
# BETA = 0.05   -> weighted experiment
# BETA = 0.1    -> stronger weighted experiment
BETA = 0.5

# When True, an existing checkpoint for this beta and seed is resumed.
# When False, the script refuses to overwrite an existing run.
RESUME_IF_AVAILABLE = True


BETA_LABEL = str(BETA).replace(".", "p")
RUN_LABEL = (
    f"beta_{BETA_LABEL}_transitions_{TOTAL_TRANSITIONS}"
    f"_seed_{RANDOM_SEED}"
)
RUN_RESULTS_DIR = RESULTS_DIR / RUN_LABEL
RUN_CHECKPOINT_DIR = CHECKPOINTS_DIR / RUN_LABEL
RUN_FIGURES_DIR = FIGURES_DIR / RUN_LABEL

for directory in (RUN_RESULTS_DIR, RUN_CHECKPOINT_DIR, RUN_FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT_PATH = RUN_CHECKPOINT_DIR / "latest_checkpoint.pkl"
LOG_PATH = RUN_RESULTS_DIR / f"run_log_{RUN_LABEL}.txt"


class Tee:
    """Write printed output to both the console and a text file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class MetropolisStyleAcceptance:
    """
    Apply alpha = min(1, exp[-beta * (C_proposed - C_current)]),
    where C is the number of cut edges.

    This is Metropolis-style rather than exact Metropolis-Hastings because
    the ReCom proposal-density ratio is not included.
    """

    def __init__(self, beta):
        self.beta = beta

    def decide(self, current_partition, proposed_partition):
        current_cuts = len(current_partition["cut_edges"])
        proposed_cuts = len(proposed_partition["cut_edges"])
        difference = proposed_cuts - current_cuts

        if difference <= 0:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp(-self.beta * difference)

        random_draw = random.random()
        accepted = random_draw < acceptance_probability

        return {
            "current_cuts": current_cuts,
            "proposed_cuts": proposed_cuts,
            "cut_difference": difference,
            "acceptance_probability": acceptance_probability,
            "random_draw": random_draw,
            "accepted": accepted,
        }


def assignment_dict(partition):
    """Return a JSON-serializable district assignment."""

    return {
        str(node): str(district)
        for node, district in partition.assignment.items()
    }


def assignment_signature(partition):
    """Return a district-label-sensitive assignment signature."""

    return tuple(
        sorted(
            (str(node), str(district))
            for node, district in partition.assignment.items()
        )
    )


def boundary_signature(partition):
    """Return a district-label-insensitive boundary signature."""

    return frozenset(
        frozenset((str(node_a), str(node_b)))
        for node_a, node_b in partition["cut_edges"]
    )


def atomic_pickle_dump(data, output_path):
    """Write a pickle atomically so an interruption cannot corrupt the old file."""

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(output_path)


def atomic_json_dump(data, output_path):
    """Write JSON atomically."""

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(output_path)


def print_partition_summary(partition, title, ideal_population):
    """Print population and cut-edge information for one partition."""

    print(f"\n===== {title} =====")
    print(f"Number of districts: {len(partition.parts)}")
    print(f"Total population: {sum(partition['population'].values()):,}")
    print(f"Ideal district population: {ideal_population:,.2f}")
    print(f"Number of cut edges: {len(partition['cut_edges'])}")

    print("\nDistrict populations:")
    for district, population in sorted(
        partition["population"].items(),
        key=lambda item: str(item[0]),
    ):
        deviation = (
            (population - ideal_population) / ideal_population * 100
        )
        print(
            f"District {district}: {population:,} "
            f"({deviation:+.3f}%)"
        )


def get_node_positions(graph):
    """Find usable node coordinates if they are present in the JSON graph."""

    coordinate_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("lon", "lat"),
        ("LON", "LAT"),
        ("INTPTLON10", "INTPTLAT10"),
    ]

    for x_name, y_name in coordinate_pairs:
        positions = {}
        try:
            for node in graph.nodes:
                positions[node] = (
                    float(graph.nodes[node][x_name]),
                    float(graph.nodes[node][y_name]),
                )
        except (KeyError, TypeError, ValueError):
            continue
        return positions

    return None


def save_partition_map(partition, state_number, label):
    """Save a map when the graph contains node coordinates."""

    positions = get_node_positions(partition.graph)
    if positions is None:
        print(
            f"Map for state {state_number} was skipped because no supported "
            "node-coordinate columns were found."
        )
        return

    district_values = sorted(
        set(partition.assignment.values()),
        key=str,
    )
    district_to_number = {
        district: index
        for index, district in enumerate(district_values)
    }
    node_colours = [
        district_to_number[partition.assignment[node]]
        for node in partition.graph.nodes
    ]

    output_path = (
        RUN_FIGURES_DIR
        / f"map_{label}_state_{state_number:05d}_{RUN_LABEL}.png"
    )

    x_coordinates = [
        positions[node][0]
        for node in partition.graph.nodes
    ]
    y_coordinates = [
        positions[node][1]
        for node in partition.graph.nodes
    ]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_coordinates,
        y_coordinates,
        c=node_colours,
        s=1.0,
        cmap="tab20",
        linewidths=0,
    )
    plt.title(
        f"Pennsylvania partition: state {state_number}\n"
        f"beta={BETA}, seed={RANDOM_SEED}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_path}")


def make_partition(graph, assignment, updaters):
    """Build a Partition from a saved assignment."""

    return Partition(
        graph=graph,
        assignment=assignment,
        updaters=updaters,
    )


def load_or_create_state(graph, initial_partition, updaters):
    """Resume the latest completed chunk or create a new run."""

    if LATEST_CHECKPOINT_PATH.exists():
        if not RESUME_IF_AVAILABLE:
            raise FileExistsError(
                f"Checkpoint already exists: {LATEST_CHECKPOINT_PATH}"
            )

        with LATEST_CHECKPOINT_PATH.open("rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        expected_settings = {
            "beta": BETA,
            "random_seed": RANDOM_SEED,
            "total_transitions": TOTAL_TRANSITIONS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "population_tolerance": POPULATION_TOLERANCE,
        }
        if checkpoint["settings"] != expected_settings:
            raise ValueError(
                "Checkpoint settings do not match the current experiment."
            )

        current_partition = make_partition(
            graph,
            checkpoint["assignment"],
            updaters,
        )
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])

        print(
            f"Resuming from transition "
            f"{checkpoint['completed_transitions']:,}."
        )
        return {
            "partition": current_partition,
            "completed_transitions": checkpoint["completed_transitions"],
            "cumulative": checkpoint["cumulative"],
            "previous_assignment_signature": assignment_signature(
                current_partition
            ),
            "previous_boundary_signature": boundary_signature(
                current_partition
            ),
        }

    existing_chunks = list(RUN_RESULTS_DIR.glob("transitions_*.csv.gz"))
    if existing_chunks:
        raise FileExistsError(
            "Transition files exist but latest_checkpoint.pkl is missing. "
            "Move this run directory aside or restore its checkpoint before "
            "starting again."
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    initial_state_path = RUN_RESULTS_DIR / "initial_state.json"
    atomic_json_dump(
        {
            "state": 0,
            "cut_edges": len(initial_partition["cut_edges"]),
            "assignment": assignment_dict(initial_partition),
        },
        initial_state_path,
    )
    save_partition_map(initial_partition, 0, "initial")

    return {
        "partition": initial_partition,
        "completed_transitions": 0,
        "cumulative": {
            "accepted": 0,
            "metropolis_rejected": 0,
            "constraint_rejected": 0,
            "assignment_repeated": 0,
            "boundary_repeated": 0,
            "minimum_cut_edges": len(initial_partition["cut_edges"]),
            "maximum_cut_edges": len(initial_partition["cut_edges"]),
        },
        "previous_assignment_signature": assignment_signature(
            initial_partition
        ),
        "previous_boundary_signature": boundary_signature(
            initial_partition
        ),
    }


def write_chunk(records, assignments, start_transition, end_transition):
    """Save one completed chunk of transition metrics and assignments."""

    stem = f"{start_transition:05d}_{end_transition:05d}"
    metrics_path = RUN_RESULTS_DIR / f"transitions_{stem}.csv.gz"
    assignments_path = RUN_RESULTS_DIR / f"assignments_{stem}.jsonl.gz"
    temporary_metrics = metrics_path.with_suffix(".csv.gz.tmp")
    temporary_assignments = assignments_path.with_suffix(".jsonl.gz.tmp")

    fieldnames = [
        "transition",
        "state",
        "cut_edges",
        "current_cuts",
        "proposed_cuts",
        "cut_difference",
        "proposal_valid",
        "accepted",
        "rejection_reason",
        "acceptance_probability",
        "random_draw",
        "assignment_repeated",
        "boundary_repeated",
    ]

    with gzip.open(temporary_metrics, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with gzip.open(temporary_assignments, "wt", encoding="utf-8") as f:
        for item in assignments:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

    temporary_metrics.replace(metrics_path)
    temporary_assignments.replace(assignments_path)

    print(f"Transition data saved to: {metrics_path}")
    print(f"Assignments saved to: {assignments_path}")


def save_checkpoint(current_partition, completed_transitions, cumulative):
    """Save both a numbered checkpoint and the latest resumable checkpoint."""

    checkpoint = {
        "settings": {
            "beta": BETA,
            "random_seed": RANDOM_SEED,
            "total_transitions": TOTAL_TRANSITIONS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "population_tolerance": POPULATION_TOLERANCE,
        },
        "completed_transitions": completed_transitions,
        "assignment": dict(current_partition.assignment),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "cumulative": cumulative,
    }

    numbered_path = (
        RUN_CHECKPOINT_DIR
        / f"checkpoint_{completed_transitions:05d}.pkl"
    )
    atomic_pickle_dump(checkpoint, numbered_path)
    atomic_pickle_dump(checkpoint, LATEST_CHECKPOINT_PATH)

    print(f"Checkpoint saved to: {numbered_path}")
    save_partition_map(
        current_partition,
        completed_transitions,
        "checkpoint",
    )


def run_chunk(
    state,
    proposal,
    constraints,
    acceptance_rule,
    chunk_end,
):
    """Run transitions up to one checkpoint boundary."""

    current_partition = state["partition"]
    completed = state["completed_transitions"]
    cumulative = state["cumulative"]
    previous_assignment = state["previous_assignment_signature"]
    previous_boundary = state["previous_boundary_signature"]

    records = []
    assignments = []

    for transition in range(completed + 1, chunk_end + 1):
        proposed_partition = proposal(current_partition)
        proposal_valid = all(
            constraint(proposed_partition)
            for constraint in constraints
        )

        if proposal_valid:
            decision = acceptance_rule.decide(
                current_partition,
                proposed_partition,
            )
            accepted = decision["accepted"]
            if accepted:
                current_partition = proposed_partition
                rejection_reason = ""
                cumulative["accepted"] += 1
            else:
                rejection_reason = "metropolis"
                cumulative["metropolis_rejected"] += 1
        else:
            current_cuts = len(current_partition["cut_edges"])
            proposed_cuts = len(proposed_partition["cut_edges"])
            decision = {
                "current_cuts": current_cuts,
                "proposed_cuts": proposed_cuts,
                "cut_difference": proposed_cuts - current_cuts,
                "acceptance_probability": "",
                "random_draw": "",
                "accepted": False,
            }
            accepted = False
            rejection_reason = "constraint"
            cumulative["constraint_rejected"] += 1

        current_assignment = assignment_signature(current_partition)
        current_boundary = boundary_signature(current_partition)
        assignment_repeated = current_assignment == previous_assignment
        boundary_repeated = current_boundary == previous_boundary

        if assignment_repeated:
            cumulative["assignment_repeated"] += 1
        if boundary_repeated:
            cumulative["boundary_repeated"] += 1

        cut_count = len(current_partition["cut_edges"])
        cumulative["minimum_cut_edges"] = min(
            cumulative["minimum_cut_edges"],
            cut_count,
        )
        cumulative["maximum_cut_edges"] = max(
            cumulative["maximum_cut_edges"],
            cut_count,
        )

        records.append(
            {
                "transition": transition,
                "state": transition,
                "cut_edges": cut_count,
                "current_cuts": decision["current_cuts"],
                "proposed_cuts": decision["proposed_cuts"],
                "cut_difference": decision["cut_difference"],
                "proposal_valid": proposal_valid,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "acceptance_probability": decision[
                    "acceptance_probability"
                ],
                "random_draw": decision["random_draw"],
                "assignment_repeated": assignment_repeated,
                "boundary_repeated": boundary_repeated,
            }
        )
        assignments.append(
            {
                "transition": transition,
                "state": transition,
                "assignment": assignment_dict(current_partition),
            }
        )

        previous_assignment = current_assignment
        previous_boundary = current_boundary

        if transition % 100 == 0 or transition == chunk_end:
            print(
                f"Completed transition {transition:,}/"
                f"{TOTAL_TRANSITIONS:,}; cut edges = {cut_count}"
            )

    return {
        "partition": current_partition,
        "completed_transitions": chunk_end,
        "cumulative": cumulative,
        "previous_assignment_signature": previous_assignment,
        "previous_boundary_signature": previous_boundary,
        "records": records,
        "assignments": assignments,
    }


def read_all_metrics():
    """Read all completed metric chunks in transition order."""

    records = []
    for path in sorted(RUN_RESULTS_DIR.glob("transitions_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            records.extend(csv.DictReader(f))
    return records


def save_cut_edge_figures(initial_cut_edges):
    """Create trace and histogram figures from all completed chunks."""

    records = read_all_metrics()
    transitions = [0] + [int(record["transition"]) for record in records]
    cut_counts = [initial_cut_edges] + [
        int(record["cut_edges"])
        for record in records
    ]

    trace_path = RUN_FIGURES_DIR / f"cut_edge_trace_{RUN_LABEL}.png"
    plt.figure(figsize=(11, 6))
    plt.plot(transitions, cut_counts, color="#315c8a", linewidth=0.8)
    plt.axhline(
        initial_cut_edges,
        color="#b4443c",
        linestyle="--",
        linewidth=1.2,
        label=f"Initial plan: {initial_cut_edges}",
    )
    plt.xlabel("Transition / resulting state")
    plt.ylabel("Number of cut edges")
    plt.title(
        "Pennsylvania weighted ReCom cut-edge trace\n"
        f"beta={BETA}, transitions={TOTAL_TRANSITIONS}, "
        f"seed={RANDOM_SEED}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(trace_path, dpi=300, bbox_inches="tight")
    plt.close()

    histogram_path = (
        RUN_FIGURES_DIR / f"cut_edge_histogram_{RUN_LABEL}.png"
    )
    plt.figure(figsize=(9, 6))
    plt.hist(cut_counts, bins=40, color="#769ac1", edgecolor="white")
    plt.axvline(
        initial_cut_edges,
        color="#b4443c",
        linestyle="--",
        linewidth=1.5,
        label=f"Initial plan: {initial_cut_edges}",
    )
    plt.xlabel("Number of cut edges")
    plt.ylabel("Number of states")
    plt.title(
        "Pennsylvania weighted ReCom cut-edge distribution\n"
        f"beta={BETA}, transitions={TOTAL_TRANSITIONS}, "
        f"seed={RANDOM_SEED}"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {trace_path}")
    print(f"Figure saved to: {histogram_path}")


def save_final_summary(
    initial_partition,
    final_partition,
    cumulative,
    ideal_population,
):
    """Save machine-readable and printed final results."""

    total = TOTAL_TRANSITIONS
    rejected = (
        cumulative["metropolis_rejected"]
        + cumulative["constraint_rejected"]
    )
    summary = {
        "beta": BETA,
        "random_seed": RANDOM_SEED,
        "initial_state": 0,
        "total_transitions": total,
        "total_states_including_initial": total + 1,
        "accepted_transitions": cumulative["accepted"],
        "rejected_transitions": rejected,
        "metropolis_rejected": cumulative["metropolis_rejected"],
        "constraint_rejected": cumulative["constraint_rejected"],
        "acceptance_rate_percent": cumulative["accepted"] / total * 100,
        "assignment_repeated": cumulative["assignment_repeated"],
        "assignment_repeat_rate_percent": (
            cumulative["assignment_repeated"] / total * 100
        ),
        "boundary_repeated": cumulative["boundary_repeated"],
        "boundary_repeat_rate_percent": (
            cumulative["boundary_repeated"] / total * 100
        ),
        "initial_cut_edges": len(initial_partition["cut_edges"]),
        "final_cut_edges": len(final_partition["cut_edges"]),
        "minimum_cut_edges": cumulative["minimum_cut_edges"],
        "maximum_cut_edges": cumulative["maximum_cut_edges"],
    }
    atomic_json_dump(
        summary,
        RUN_RESULTS_DIR / f"summary_{RUN_LABEL}.json",
    )
    atomic_json_dump(
        {
            "state": TOTAL_TRANSITIONS,
            "cut_edges": len(final_partition["cut_edges"]),
            "assignment": assignment_dict(final_partition),
        },
        RUN_RESULTS_DIR / "final_state.json",
    )

    print("\n===== Final Results =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print_partition_summary(
        final_partition,
        "Final Pennsylvania Partition",
        ideal_population,
    )


def run_experiment():
    """Load the graph, run or resume the chain, and save all outputs."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Pennsylvania graph file was not found: {DATA_PATH}"
        )

    print("Loading Pennsylvania graph...")
    graph = Graph.from_json(str(DATA_PATH))
    print("Graph loaded successfully.")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")

    missing_population = [
        node
        for node in graph.nodes
        if graph.nodes[node].get(POPULATION_COLUMN) is None
    ]
    missing_district = [
        node
        for node in graph.nodes
        if graph.nodes[node].get(DISTRICT_COLUMN) is None
    ]
    if missing_population:
        raise ValueError(
            f"{len(missing_population)} nodes are missing "
            f"{POPULATION_COLUMN}."
        )
    if missing_district:
        raise ValueError(
            f"{len(missing_district)} nodes are missing "
            f"{DISTRICT_COLUMN}."
        )

    updaters = {
        "population": Tally(POPULATION_COLUMN, alias="population"),
        "cut_edges": cut_edges,
    }
    initial_partition = Partition(
        graph=graph,
        assignment=DISTRICT_COLUMN,
        updaters=updaters,
    )

    if len(initial_partition.parts) != NUMBER_OF_DISTRICTS:
        raise ValueError(
            f"Expected {NUMBER_OF_DISTRICTS} districts, "
            f"but found {len(initial_partition.parts)}."
        )

    total_population = sum(initial_partition["population"].values())
    ideal_population = total_population / NUMBER_OF_DISTRICTS
    print_partition_summary(
        initial_partition,
        "Initial Pennsylvania Partition",
        ideal_population,
    )

    proposal = partial(
        recom,
        pop_col=POPULATION_COLUMN,
        pop_target=ideal_population,
        epsilon=POPULATION_TOLERANCE,
        node_repeats=1,
    )
    constraints = [
        contiguous,
        within_percent_of_ideal_population(
            initial_partition,
            POPULATION_TOLERANCE,
        ),
    ]
    acceptance_rule = MetropolisStyleAcceptance(BETA)

    print("\n===== Experiment Settings =====")
    print(f"Beta: {BETA}")
    print(f"Total transitions: {TOTAL_TRANSITIONS}")
    print(f"Total states including initial: {TOTAL_TRANSITIONS + 1}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"Population tolerance: "
        f"{POPULATION_TOLERANCE * 100:.1f}%"
    )

    state = load_or_create_state(
        graph,
        initial_partition,
        updaters,
    )

    while state["completed_transitions"] < TOTAL_TRANSITIONS:
        chunk_start = state["completed_transitions"] + 1
        chunk_end = min(
            state["completed_transitions"] + CHECKPOINT_INTERVAL,
            TOTAL_TRANSITIONS,
        )
        print(
            f"\nRunning transitions {chunk_start:,} to "
            f"{chunk_end:,}..."
        )

        completed_chunk = run_chunk(
            state,
            proposal,
            constraints,
            acceptance_rule,
            chunk_end,
        )
        write_chunk(
            completed_chunk["records"],
            completed_chunk["assignments"],
            chunk_start,
            chunk_end,
        )
        save_checkpoint(
            completed_chunk["partition"],
            completed_chunk["completed_transitions"],
            completed_chunk["cumulative"],
        )

        state = {
            key: value
            for key, value in completed_chunk.items()
            if key not in ("records", "assignments")
        }

    final_partition = state["partition"]
    save_partition_map(
        final_partition,
        TOTAL_TRANSITIONS,
        "final",
    )
    save_cut_edge_figures(len(initial_partition["cut_edges"]))
    save_final_summary(
        initial_partition,
        final_partition,
        state["cumulative"],
        ideal_population,
    )
    print("\nRun completed successfully.")


def main():
    log_mode = "a" if LATEST_CHECKPOINT_PATH.exists() else "w"
    original_stdout = sys.stdout

    with LOG_PATH.open(log_mode, encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            run_experiment()
        finally:
            sys.stdout = original_stdout

    print(f"Run log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
