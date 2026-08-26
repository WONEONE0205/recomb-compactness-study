from pathlib import Path

from gerrychain import Graph


# --------------------------------------------------
# File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

GRAPH_FILE = DATA_DIR / "PA_VTDs.json"


# --------------------------------------------------
# Data columns
# --------------------------------------------------

POPULATION_COLUMN = "TOT_POP"
DISTRICT_COLUMN = "2011_PLA_1"

VOTE_COLUMNS = [
    "PRES12D",
    "PRES12R",
    "T16PRESD",
    "T16PRESR",
]


# --------------------------------------------------
# Load graph
# --------------------------------------------------

def load_graph() -> Graph:
    """Load the Pennsylvania graph."""

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found:\n{GRAPH_FILE}\n"
            "Please check the file name in the data directory."
        )

    print(f"Loading graph from:\n{GRAPH_FILE}\n")

    graph = Graph.from_json(str(GRAPH_FILE))

    return graph


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def count_nodes_with_attribute(graph: Graph, attribute: str) -> int:
    """Count nodes that contain a given attribute."""

    return sum(
        1
        for node in graph.nodes
        if graph.nodes[node].get(attribute) is not None
    )


def sum_attribute(graph: Graph, attribute: str) -> float:
    """Sum the values of a node attribute."""

    return sum(
        graph.nodes[node].get(attribute, 0) or 0
        for node in graph.nodes
    )


# --------------------------------------------------
# Inspect graph
# --------------------------------------------------

def inspect_graph(graph: Graph) -> None:
    """Print basic information about the graph."""

    print("===== Graph Summary =====")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")

    if len(graph.nodes) == 0:
        raise ValueError("The graph contains no nodes.")

    first_node = next(iter(graph.nodes))
    available_columns = list(graph.nodes[first_node].keys())

    print("\n===== Example Node =====")
    print(f"Node ID: {first_node}")
    print(f"Available columns: {available_columns}")

    print("\n===== Population Check =====")

    population_nodes = count_nodes_with_attribute(
        graph,
        POPULATION_COLUMN,
    )

    total_population = sum_attribute(
        graph,
        POPULATION_COLUMN,
    )

    print(
        f"Nodes with {POPULATION_COLUMN}: "
        f"{population_nodes}/{len(graph.nodes)}"
    )

    print(f"Total population: {total_population:,.0f}")

    print("\n===== District Check =====")

    district_values = {
        graph.nodes[node].get(DISTRICT_COLUMN)
        for node in graph.nodes
        if graph.nodes[node].get(DISTRICT_COLUMN) is not None
    }

    print(f"District column: {DISTRICT_COLUMN}")
    print(f"Number of districts: {len(district_values)}")
    print(f"District labels: {sorted(district_values)}")

    print("\n===== Election Data Check =====")

    for column in VOTE_COLUMNS:
        node_count = count_nodes_with_attribute(graph, column)
        total_votes = sum_attribute(graph, column)

        print(
            f"{column}: "
            f"{node_count}/{len(graph.nodes)} nodes, "
            f"total = {total_votes:,.0f}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    graph = load_graph()
    inspect_graph(graph)

    print("\nData loaded successfully.")


if __name__ == "__main__":
    main()
