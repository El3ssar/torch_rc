import networkx as nx

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology("zeros")
def zero_graph(n: int) -> nx.Graph:
    """
    Create the zero (edgeless) graph with n nodes.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    nx.Graph
        Graph with n nodes and zero edges.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    return G
