# main.py
# FastAPI backend - wires together the full Codebase Knowledge AI pipeline:
# clone -> parse -> build graph -> index -> retrieve -> answer

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from repo_parser import parse_repo
from graph_builder import build_graph, save_graph, load_graph
from indexer import index_repo
from retriever import answer_question

app = FastAPI()

# In-memory cache for the current session - avoids re-loading from disk
# every single time a question is asked about the same repo
session_cache = {}


class AnalyzeRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    repo_name: str
    question: str


def get_repo_name_from_url(repo_url: str) -> str:
    # Extracts a clean repo name from a GitHub URL
    # e.g. https://github.com/user/my-repo -> "my-repo"
    name = repo_url.rstrip("/").split("/")[-1]
    return name.replace(".git", "")


def get_metadata_path(repo_name: str) -> str:
    return os.path.join("repo_cache", repo_name, "metadata.json")


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    repo_name = get_repo_name_from_url(request.repo_url)

    try:
        # Check if we've already analyzed this repo before
        cached_graph = load_graph(repo_name)

        if cached_graph is not None:
            # Already analyzed - just reload parsed files for source context
            # (parsed_files aren't persisted separately, so re-parse from the cached clone)
            from repo_parser import find_python_files, parse_python_file

            repo_path = os.path.join("cloned_repos", repo_name)
            python_files = find_python_files(repo_path)
            parsed_files = [parse_python_file(f) for f in python_files]
            parsed_files = [f for f in parsed_files if f]

            session_cache[repo_name] = {
                "graph": cached_graph,
                "parsed_files": parsed_files
            }

            return {
                "repo_name": repo_name,
                "status": "loaded_from_cache",
                "file_count": len(parsed_files),
                "node_count": cached_graph.number_of_nodes()
            }

        # Not cached - run the full pipeline
        parsed_files = parse_repo(request.repo_url, repo_name)

        if not parsed_files:
            raise HTTPException(status_code=400, detail="No Python files found in this repo.")

        graph = build_graph(parsed_files)
        save_graph(graph, repo_name)

        vector_count = index_repo(graph, repo_name)

        # Save metadata for future reference
        os.makedirs(os.path.join("repo_cache", repo_name), exist_ok=True)
        with open(get_metadata_path(repo_name), "w", encoding="utf-8") as f:
            json.dump({
                "repo_url": request.repo_url,
                "file_count": len(parsed_files),
                "node_count": graph.number_of_nodes(),
                "vector_count": vector_count
            }, f)

        session_cache[repo_name] = {
            "graph": graph,
            "parsed_files": parsed_files
        }

        return {
            "repo_name": repo_name,
            "status": "newly_analyzed",
            "file_count": len(parsed_files),
            "node_count": graph.number_of_nodes(),
            "vector_count": vector_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")


@app.post("/ask")
async def ask(request: AskRequest):
    if request.repo_name not in session_cache:
        raise HTTPException(
            status_code=400,
            detail="This repo hasn't been analyzed yet. Call /analyze first."
        )

    try:
        cached = session_cache[request.repo_name]
        result = answer_question(
            question=request.question,
            repo_name=request.repo_name,
            graph=cached["graph"],
            parsed_files=cached["parsed_files"]
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


# Mount static files last to avoid conflicting with API routes
app.mount("/static", StaticFiles(directory="static"), name="static")