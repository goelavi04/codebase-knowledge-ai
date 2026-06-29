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


def is_summary_question(question: str) -> bool:
    """
    Detects broad 'tell me about this repo' style questions
    that need wide context instead of narrow targeted search.
    """
    summary_keywords = [
        "what is this repo", "what does this repo", "what is this project",
        "summary", "summarize", "overview", "what is the repo about",
        "what does this do", "explain this repo", "explain this project",
        "what's this about"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summary_keywords)


def hybrid_retrieve(question: str, repo_name: str, graph, parsed_files: list) -> dict:
    """
    Step 1: Vector search for semantically relevant functions/classes
             (wider net for summary-style questions)
    Step 2: Graph traversal to pull in connected nodes for each result
    Step 3: Build combined context for the LLM
    """
    # Step 1 - vector search - widen the net for summary-style questions
    top_k = 20 if is_summary_question(question) else 5
    vector_matches = search_repo(question, repo_name, top_k=top_k)

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
    and returns a plain-English, well-formatted answer.
    """
    system_prompt = (
        "You are a senior software engineer explaining a codebase to a teammate. "
        "Use the provided context about functions, classes, and their relationships "
        "to answer the question clearly and specifically.\n\n"
        "Formatting rules:\n"
        "- Use short bullet points for lists of files, functions, or features\n"
        "- Use a brief intro sentence before bullets, not just a bullet dump\n"
        "- Bold key file names and function names using **name** style\n"
        "- Keep bullets concise - one idea per bullet\n"
        "- If asked for a general summary or overview of the repo, synthesize "
        "across ALL the files and functions in the context into a coherent "
        "paragraph first, THEN list key components as bullets - don't just "
        "list isolated functions without explaining the bigger picture\n"
        "- If the context doesn't contain enough information, say so honestly "
        "rather than guessing"
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