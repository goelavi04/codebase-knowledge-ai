# graph_builder.py
# Converts parsed repo data into a NetworkX graph representing code relationships
# Also handles saving/loading the graph to/from disk for persistence

import os
import pickle
import networkx as nx


def build_graph(parsed_files: list) -> nx.DiGraph:
    """
    Builds a directed graph from parsed file data.
    Nodes: files, functions, classes
    Edges: imports, calls, defined_in
    """
    graph = nx.DiGraph()

    # First pass - create all nodes
    for file_data in parsed_files:
        file_path = file_data["file_path"]
        file_name = os.path.basename(file_path)

        # Add the file itself as a node
        graph.add_node(file_path, type="file", name=file_name)

        # Add each function as a node, linked to its file
        for func in file_data["functions"]:
            func_node_id = f"{file_path}::{func['name']}"
            graph.add_node(
                func_node_id,
                type="function",
                name=func["name"],
                docstring=func["docstring"],
                file_path=file_path
            )
            # Edge: function is defined in this file
            graph.add_edge(func_node_id, file_path, relation="defined_in")

        # Add each class as a node, linked to its file
        for cls in file_data["classes"]:
            cls_node_id = f"{file_path}::{cls['name']}"
            graph.add_node(
                cls_node_id,
                type="class",
                name=cls["name"],
                docstring=cls["docstring"],
                file_path=file_path
            )
            graph.add_edge(cls_node_id, file_path, relation="defined_in")

    # Second pass - create edges for imports and function calls
    # (done in a second pass because we need all nodes to exist first)
    file_lookup = {
        os.path.basename(f["file_path"]).replace(".py", ""): f["file_path"]
        for f in parsed_files
    }

    for file_data in parsed_files:
        file_path = file_data["file_path"]

        # Edge: this file imports another file in the repo
        for imported_module in file_data["imports"]:
            module_name = imported_module.split(".")[-1]
            if module_name in file_lookup:
                target_file = file_lookup[module_name]
                if target_file != file_path:
                    graph.add_edge(file_path, target_file, relation="imports")

        # Edge: this function calls another function we know about
        for func in file_data["functions"]:
            func_node_id = f"{file_path}::{func['name']}"

            for called_name in func["calls"]:
                # Search all functions across the repo for a name match
                for other_file in parsed_files:
                    for other_func in other_file["functions"]:
                        if other_func["name"] == called_name:
                            target_node_id = f"{other_file['file_path']}::{called_name}"
                            if target_node_id != func_node_id:
                                graph.add_edge(func_node_id, target_node_id, relation="calls")

    return graph


def save_graph(graph: nx.DiGraph, repo_name: str):
    """
    Saves the graph to disk as a pickle file under repo_cache/{repo_name}/graph.pickle
    """
    cache_dir = os.path.join("repo_cache", repo_name)
    os.makedirs(cache_dir, exist_ok=True)

    graph_path = os.path.join(cache_dir, "graph.pickle")
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(repo_name: str) -> nx.DiGraph:
    """
    Loads a previously saved graph from disk.
    Returns None if no cached graph exists for this repo.
    """
    graph_path = os.path.join("repo_cache", repo_name, "graph.pickle")

    if not os.path.exists(graph_path):
        return None

    with open(graph_path, "rb") as f:
        return pickle.load(f)


def get_connected_nodes(graph: nx.DiGraph, node_id: str, depth: int = 1) -> list:
    """
    Given a node (file or function), finds all nodes connected to it
    up to a certain depth - both what it depends ON and what depends ON it.
    """
    if node_id not in graph:
        return []

    connected = set()

    # Nodes this node points to (its dependencies)
    descendants = nx.descendants_at_distance(graph, node_id, depth) if depth else set()
    connected.update(descendants)

    # Nodes that point to this node (things that depend on it)
    reversed_graph = graph.reverse()
    ancestors = nx.descendants_at_distance(reversed_graph, node_id, depth) if depth else set()
    connected.update(ancestors)

    return list(connected)