from pathlib import Path
import random

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

ROWS = 8
COLS = 8
NODE_POPULATION = 100
NUMBER_OF_DISTRICTS = 4
POPULATION_TOLERANCE = 0.01
RANDOM_SEED = 42
MAX_TREE_ATTEMPTS = 10000

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = Path(r"D:\python project\Gerrymandering\figures")
OUTPUT_FILE = FIGURES_DIR / "toy_recom_proposal_process.png"

DISTRICT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
]

MERGED_COLOR = "#B9A7D0"
UNSELECTED_COLOR = "#D9D9D9"


# ============================================================
# Graph and initial partition
# ============================================================

def create_grid_graph(rows, cols):
    """Create a grid graph with equal population at every node."""

    graph = nx.grid_2d_graph(rows, cols)

    for node in graph.nodes:
        graph.nodes[node]["population"] = NODE_POPULATION

    return graph


def create_initial_assignment(rows, cols):
    """Divide the grid into four equal rectangular districts."""

    assignment = {}

    row_midpoint = rows // 2
    column_midpoint = cols // 2

    for row in range(rows):
        for column in range(cols):
            node = (row, column)

            if row < row_midpoint and column < column_midpoint:
                assignment[node] = 1
            elif row < row_midpoint and column >= column_midpoint:
                assignment[node] = 2
            elif row >= row_midpoint and column < column_midpoint:
                assignment[node] = 3
            else:
                assignment[node] = 4

    return assignment


def create_positions(graph, rows):
    """Create positions with the first grid row displayed at the top."""

    return {
        (row, column): (column, rows - 1 - row)
        for row, column in graph.nodes
    }


# ============================================================
# District calculations
# ============================================================

def get_district_nodes(assignment, district):
    """Return all nodes assigned to a specified district."""

    return {
        node
        for node, district_label in assignment.items()
        if district_label == district
    }


def calculate_population(graph, nodes):
    """Calculate the total population of a collection of nodes."""

    return sum(
        graph.nodes[node]["population"]
        for node in nodes
    )


def find_adjacent_district_pairs(graph, assignment):
    """Find all pairs of districts sharing at least one graph edge."""

    adjacent_pairs = set()

    for node_a, node_b in graph.edges:
        district_a = assignment[node_a]
        district_b = assignment[node_b]

        if district_a != district_b:
            adjacent_pairs.add(tuple(sorted((district_a, district_b))))

    return sorted(adjacent_pairs)


# ============================================================
# Random spanning tree
# ============================================================

class DisjointSet:
    """A small disjoint-set structure used by randomized Kruskal."""

    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])

        return self.parent[node]

    def union(self, node_a, node_b):
        root_a = self.find(node_a)
        root_b = self.find(node_b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True


def generate_random_spanning_tree(subgraph, rng):
    """Generate a random spanning tree using randomized Kruskal."""

    edges = list(subgraph.edges)
    rng.shuffle(edges)

    disjoint_set = DisjointSet(subgraph.nodes)
    tree = nx.Graph()
    tree.add_nodes_from(subgraph.nodes)

    for node_a, node_b in edges:
        if disjoint_set.union(node_a, node_b):
            tree.add_edge(node_a, node_b)

        if tree.number_of_edges() == tree.number_of_nodes() - 1:
            break

    if not nx.is_tree(tree):
        raise RuntimeError("A spanning tree could not be generated.")

    return tree


# ============================================================
# Balanced cut
# ============================================================

def find_balanced_cut_edges(
    graph,
    tree,
    ideal_population,
    population_tolerance,
):
    """Find tree edges producing two population-balanced components."""

    minimum_population = ideal_population * (1 - population_tolerance)
    maximum_population = ideal_population * (1 + population_tolerance)

    balanced_cuts = []

    for edge in list(tree.edges):
        node_a, node_b = edge

        tree.remove_edge(node_a, node_b)
        components = list(nx.connected_components(tree))
        tree.add_edge(node_a, node_b)

        if len(components) != 2:
            continue

        component_a = set(components[0])
        component_b = set(components[1])

        population_a = calculate_population(graph, component_a)
        population_b = calculate_population(graph, component_b)

        population_a_valid = (
            minimum_population
            <= population_a
            <= maximum_population
        )

        population_b_valid = (
            minimum_population
            <= population_b
            <= maximum_population
        )

        if population_a_valid and population_b_valid:
            balanced_cuts.append(
                {
                    "edge": tuple(sorted(edge)),
                    "component_a": component_a,
                    "component_b": component_b,
                    "population_a": population_a,
                    "population_b": population_b,
                }
            )

    return balanced_cuts


def cut_reproduces_original_districts(
    component_a,
    component_b,
    original_district_a_nodes,
    original_district_b_nodes,
):
    """Check whether a cut simply recreates the original two districts."""

    original_pair = {
        frozenset(original_district_a_nodes),
        frozenset(original_district_b_nodes),
    }

    proposed_pair = {
        frozenset(component_a),
        frozenset(component_b),
    }

    return proposed_pair == original_pair


def generate_recom_proposal(
    graph,
    assignment,
    selected_pair,
    rng,
):
    """Generate one population-balanced ReCom proposal."""

    district_a, district_b = selected_pair

    district_a_nodes = get_district_nodes(
        assignment,
        district_a,
    )

    district_b_nodes = get_district_nodes(
        assignment,
        district_b,
    )

    merged_nodes = district_a_nodes | district_b_nodes
    merged_subgraph = graph.subgraph(merged_nodes).copy()

    merged_population = calculate_population(
        graph,
        merged_nodes,
    )

    ideal_population = merged_population / 2

    fallback_result = None

    for attempt in range(1, MAX_TREE_ATTEMPTS + 1):
        spanning_tree = generate_random_spanning_tree(
            merged_subgraph,
            rng,
        )

        balanced_cuts = find_balanced_cut_edges(
            graph,
            spanning_tree,
            ideal_population,
            POPULATION_TOLERANCE,
        )

        if not balanced_cuts:
            continue

        rng.shuffle(balanced_cuts)

        for balanced_cut in balanced_cuts:
            result = {
                "district_a": district_a,
                "district_b": district_b,
                "district_a_nodes": district_a_nodes,
                "district_b_nodes": district_b_nodes,
                "merged_nodes": merged_nodes,
                "merged_subgraph": merged_subgraph,
                "merged_population": merged_population,
                "ideal_population": ideal_population,
                "spanning_tree": spanning_tree,
                "cut_edge": balanced_cut["edge"],
                "component_a": balanced_cut["component_a"],
                "component_b": balanced_cut["component_b"],
                "population_a": balanced_cut["population_a"],
                "population_b": balanced_cut["population_b"],
                "tree_attempts": attempt,
            }

            if fallback_result is None:
                fallback_result = result

            reproduces_original = cut_reproduces_original_districts(
                balanced_cut["component_a"],
                balanced_cut["component_b"],
                district_a_nodes,
                district_b_nodes,
            )

            if not reproduces_original:
                return result

    if fallback_result is not None:
        return fallback_result

    raise RuntimeError(
        "No balanced cut was found. "
        "Try changing RANDOM_SEED or increasing MAX_TREE_ATTEMPTS."
    )


# ============================================================
# Proposed assignment
# ============================================================

def component_overlap(component, district_nodes):
    """Count how many nodes a component shares with an old district."""

    return len(component & district_nodes)


def create_proposed_assignment(assignment, proposal):
    """Create a full four-district assignment from the ReCom components."""

    proposed_assignment = assignment.copy()

    district_a = proposal["district_a"]
    district_b = proposal["district_b"]

    component_a = proposal["component_a"]
    component_b = proposal["component_b"]

    old_district_a_nodes = proposal["district_a_nodes"]

    overlap_a = component_overlap(
        component_a,
        old_district_a_nodes,
    )

    overlap_b = component_overlap(
        component_b,
        old_district_a_nodes,
    )

    if overlap_a >= overlap_b:
        nodes_for_district_a = component_a
        nodes_for_district_b = component_b
    else:
        nodes_for_district_a = component_b
        nodes_for_district_b = component_a

    for node in nodes_for_district_a:
        proposed_assignment[node] = district_a

    for node in nodes_for_district_b:
        proposed_assignment[node] = district_b

    return proposed_assignment


# ============================================================
# Validation
# ============================================================

def district_is_contiguous(graph, assignment, district):
    """Check whether one district forms a connected subgraph."""

    district_nodes = get_district_nodes(
        assignment,
        district,
    )

    if not district_nodes:
        return False

    district_subgraph = graph.subgraph(district_nodes)

    return nx.is_connected(district_subgraph)


def all_districts_are_contiguous(graph, assignment):
    """Check whether every district is contiguous."""

    districts = sorted(set(assignment.values()))

    return all(
        district_is_contiguous(
            graph,
            assignment,
            district,
        )
        for district in districts
    )


def population_balance_is_satisfied(
    graph,
    proposed_assignment,
    district_a,
    district_b,
    ideal_population,
):
    """Check population balance for the two recombined districts."""

    minimum_population = (
        ideal_population * (1 - POPULATION_TOLERANCE)
    )

    maximum_population = (
        ideal_population * (1 + POPULATION_TOLERANCE)
    )

    for district in (district_a, district_b):
        district_nodes = get_district_nodes(
            proposed_assignment,
            district,
        )

        district_population = calculate_population(
            graph,
            district_nodes,
        )

        if not (
            minimum_population
            <= district_population
            <= maximum_population
        ):
            return False

    return True


# ============================================================
# Plotting
# ============================================================

def draw_base_graph(
    axis,
    graph,
    positions,
    node_colors,
    title,
):
    """Draw the grid graph using the supplied node colours."""

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edge_color="#AAAAAA",
        width=1.0,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=node_colors,
        node_size=260,
        edgecolors="white",
        linewidths=0.8,
    )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    axis.set_aspect("equal")
    axis.axis("off")


def assignment_to_colors(assignment):
    """Convert district labels into plotting colours."""

    return [
        DISTRICT_COLORS[assignment[node] - 1]
        for node in assignment
    ]


def create_process_figure(
    graph,
    assignment,
    proposed_assignment,
    proposal,
    positions,
):
    """Create the four-panel ReCom proposal diagram."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
    )

    axes = axes.flatten()

    selected_districts = {
        proposal["district_a"],
        proposal["district_b"],
    }

    # Panel A: initial partition and selected adjacent districts
    panel_a_colors = []

    for node in graph.nodes:
        district = assignment[node]

        if district in selected_districts:
            panel_a_colors.append(
                DISTRICT_COLORS[district - 1]
            )
        else:
            panel_a_colors.append(UNSELECTED_COLOR)

    draw_base_graph(
        axes[0],
        graph,
        positions,
        panel_a_colors,
        "(a) Select adjacent districts",
    )

    selected_boundary_edges = [
        (node_a, node_b)
        for node_a, node_b in graph.edges
        if {
            assignment[node_a],
            assignment[node_b],
        } == selected_districts
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axes[0],
        edgelist=selected_boundary_edges,
        edge_color="black",
        width=3.0,
    )

    # Panel B: merged region and random spanning tree
    panel_b_colors = [
        MERGED_COLOR
        if node in proposal["merged_nodes"]
        else UNSELECTED_COLOR
        for node in graph.nodes
    ]

    draw_base_graph(
        axes[1],
        graph,
        positions,
        panel_b_colors,
        "(b) Merge and build a spanning tree",
    )

    nx.draw_networkx_edges(
        proposal["spanning_tree"],
        positions,
        ax=axes[1],
        edge_color="#222222",
        width=2.1,
    )

    # Panel C: balanced cut
    component_colors = []

    for node in graph.nodes:
        if node in proposal["component_a"]:
            component_colors.append("#80B1D3")
        elif node in proposal["component_b"]:
            component_colors.append("#FDB462")
        else:
            component_colors.append(UNSELECTED_COLOR)

    draw_base_graph(
        axes[2],
        graph,
        positions,
        component_colors,
        "(c) Remove a balanced cut edge",
    )

    remaining_tree_edges = [
        edge
        for edge in proposal["spanning_tree"].edges
        if set(edge) != set(proposal["cut_edge"])
    ]

    nx.draw_networkx_edges(
        proposal["spanning_tree"],
        positions,
        ax=axes[2],
        edgelist=remaining_tree_edges,
        edge_color="#555555",
        width=1.8,
    )

    nx.draw_networkx_edges(
        proposal["spanning_tree"],
        positions,
        ax=axes[2],
        edgelist=[proposal["cut_edge"]],
        edge_color="#D62728",
        width=4.5,
    )

    # Panel D: proposed partition
    panel_d_colors = assignment_to_colors(
        proposed_assignment
    )

    draw_base_graph(
        axes[3],
        graph,
        positions,
        panel_d_colors,
        "(d) Proposed partition",
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DISTRICT_COLORS[index],
            markeredgecolor="white",
            markersize=10,
            label=f"District {index + 1}",
        )
        for index in range(NUMBER_OF_DISTRICTS)
    ]

    figure.legend(
        handles=legend_items,
        loc="lower center",
        ncol=NUMBER_OF_DISTRICTS,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    figure.suptitle(
        "Toy Model Demonstration of a ReCom Proposal",
        fontsize=16,
        fontweight="bold",
        y=1,
    )

    figure.subplots_adjust(
        left=0.06,
        right=0.94,
        bottom=0.10,
        top=0.91,
        wspace=0.12,
        hspace=0.28,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(figure)


# ============================================================
# Output
# ============================================================

def print_grid_assignment(
    assignment,
    rows,
    cols,
):
    """Print the district assignment as a grid."""

    for row in range(rows):
        labels = [
            str(assignment[(row, column)])
            for column in range(cols)
        ]

        print(" ".join(labels))


def print_results(
    graph,
    proposed_assignment,
    proposal,
    population_valid,
    contiguity_valid,
):
    """Print a short summary of the demonstrated proposal."""

    district_a = proposal["district_a"]
    district_b = proposal["district_b"]

    district_a_population = calculate_population(
        graph,
        get_district_nodes(
            proposed_assignment,
            district_a,
        ),
    )

    district_b_population = calculate_population(
        graph,
        get_district_nodes(
            proposed_assignment,
            district_b,
        ),
    )

    print("===== ReCom Proposal Demonstration =====")
    print(
        "Selected adjacent districts: "
        f"{district_a} and {district_b}"
    )
    print(
        "Population of merged region: "
        f"{proposal['merged_population']}"
    )
    print(
        "Target population for each new district: "
        f"{proposal['ideal_population']:.0f}"
    )
    print(
        "Random spanning trees attempted: "
        f"{proposal['tree_attempts']}"
    )

    print("\nBalanced cut found")
    print(
        "Removed spanning-tree edge: "
        f"{proposal['cut_edge']}"
    )
    print(
        "Population of first component: "
        f"{proposal['population_a']}"
    )
    print(
        "Population of second component: "
        f"{proposal['population_b']}"
    )

    print("\nProposed district populations")
    print(
        f"District {district_a}: "
        f"{district_a_population}"
    )
    print(
        f"District {district_b}: "
        f"{district_b_population}"
    )

    print(
        "\nProposal satisfies population balance: "
        f"{population_valid}"
    )
    print(
        "Proposal contains four contiguous districts: "
        f"{contiguity_valid}"
    )

    print("\nProposed district assignment:")
    print_grid_assignment(
        proposed_assignment,
        ROWS,
        COLS,
    )

    print(
        "\nFigure saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# Main program
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    graph = create_grid_graph(
        ROWS,
        COLS,
    )

    positions = create_positions(
        graph,
        ROWS,
    )

    initial_assignment = create_initial_assignment(
        ROWS,
        COLS,
    )

    adjacent_pairs = find_adjacent_district_pairs(
        graph,
        initial_assignment,
    )

    if not adjacent_pairs:
        raise RuntimeError(
            "The initial partition contains no adjacent districts."
        )

    selected_pair = rng.choice(adjacent_pairs)

    proposal = generate_recom_proposal(
        graph,
        initial_assignment,
        selected_pair,
        rng,
    )

    proposed_assignment = create_proposed_assignment(
        initial_assignment,
        proposal,
    )

    population_valid = population_balance_is_satisfied(
        graph,
        proposed_assignment,
        proposal["district_a"],
        proposal["district_b"],
        proposal["ideal_population"],
    )

    contiguity_valid = all_districts_are_contiguous(
        graph,
        proposed_assignment,
    )

    create_process_figure(
        graph,
        initial_assignment,
        proposed_assignment,
        proposal,
        positions,
    )

    print_results(
        graph,
        proposed_assignment,
        proposal,
        population_valid,
        contiguity_valid,
    )


if __name__ == "__main__":
    main()
