# retriever.py
# Combines vector search (Pinecone) and graph traversal (NetworkX)
# This is the core GraphRAG logic - then sends context to an LLM for the final answer

import os
import requests
from dotenv import load_dotenv
from indexer import search_repo
from graph_builder import get_connected_nodes

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"


def get_node_source_snippet(graph, node_id: str, parsed_files: list, max_lines: int = 15) -> str:
    """
    Given a function/class node, finds its file's full source code
    and returns it (we keep this simple - full file context helps the LLM).
    """
    if node_id not in graph:
        return ""

    file_path = graph.nodes[node_id].get("file_path", "")

    for file_data in parsed_files:
        if file_data["file_path"] == file_path:
            lines = file_data["source_code"].split("\n")
            return "\n".join(lines[:max_lines * 3])  # rough slice, keeps context manageable

    return ""


def hybrid_retrieve(question: str, repo_name: str, graph, parsed_files: list) -> dict:
    """
    Step 1: Vector search for semantically relevant functions/classes
    Step 2: Graph traversal to pull in connected nodes for each result
    Step 3: Build combined context for the LLM
    """
    # Step 1 - vector search
    vector_matches = search_repo(question, repo_name, top_k=5)

    all_relevant_nodes = set()
    primary_nodes = []

    for match in vector_matches:
        node_id = match["node_id"]
        primary_nodes.append(match)
        all_relevant_nodes.add(node_id)

        # Step 2 - graph traversal - pull in directly connected nodes
        connected = get_connected_nodes(graph, node_id, depth=1)
        all_relevant_nodes.update(connected)

    # Step 3 - build context text combining everything we found
    context_parts = []

    for node_id in all_relevant_nodes:
        if node_id not in graph:
            continue

        node_data = graph.nodes[node_id]
        node_type = node_data.get("type", "")
        name = node_data.get("name", "")
        docstring = node_data.get("docstring", "")
        file_path = node_data.get("file_path", node_id)

        context_parts.append(
            f"[{node_type}] {name} (in {file_path})\n"
            f"Docstring: {docstring if docstring else 'None'}\n"
        )

    context_text = "\n".join(context_parts)

    return {
        "primary_matches": primary_nodes,
        "context_text": context_text,
        "total_nodes_explored": len(all_relevant_nodes)
    }


def generate_answer(question: str, context_text: str) -> str:
    """
    Sends the question + combined GraphRAG context to OpenRouter
    and returns a plain-English answer.
    """
    system_prompt = (
        "You are a senior software engineer explaining a codebase to a teammate. "
        "Use the provided context about functions, classes, and their relationships "
        "to answer the question clearly and specifically. "
        "Reference exact file paths and function names from the context. "
        "If the context doesn't contain enough information, say so honestly."
    )

    user_prompt = f"Question: {question}\n\nCodebase context:\n{context_text}"

    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    )

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"Error generating answer: {result}"


def answer_question(question: str, repo_name: str, graph, parsed_files: list) -> dict:
    """
    Main entry point - runs the full GraphRAG pipeline and returns
    the final answer plus the sources used to generate it.
    """
    retrieval = hybrid_retrieve(question, repo_name, graph, parsed_files)
    answer = generate_answer(question, retrieval["context_text"])

    return {
        "answer": answer,
        "sources": retrieval["primary_matches"],
        "nodes_explored": retrieval["total_nodes_explored"]
    }