"""CLI: interactive chat, one-shot ask, ingest prompt, reindex."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from llm_wiki.agent import create_wiki_agent, create_wiki_agent_for_batch_ingestion
from llm_wiki.indexing import index_folder_into_knowledge, reindex_all_wiki_pages
from llm_wiki.paths import WikiPaths, ensure_layout, wiki_paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agno LLM Wiki — Karpathy-style wiki with vector index")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Wiki root (default: directory containing this package's raw/wiki/data)",
    )
    p.add_argument("--debug", action="store_true", help="Agno debug_mode on the agent")
    sub = p.add_subparsers(dest="command", required=False)

    chat = sub.add_parser("chat", help="Interactive REPL (default)")
    chat.set_defaults(_run=_cmd_chat)

    ask = sub.add_parser("ask", help="Single question then exit")
    ask.add_argument("question", nargs=argparse.REMAINDER, help="Question text")
    ask.set_defaults(_run=_cmd_ask)

    ingest = sub.add_parser("ingest", help="One-shot ingest instruction for a file in raw/")
    ingest.add_argument("filename", help="Basename of a file under raw/")
    ingest.set_defaults(_run=_cmd_ingest)

    reindex = sub.add_parser("reindex", help="Embed all wiki/**/*.md into the knowledge base")
    reindex.set_defaults(_run=_cmd_reindex)

    idx_folder = sub.add_parser(
        "index-folder",
        help="Embed files under an external folder (e.g. repo docs/) into the knowledge base",
    )
    idx_folder.add_argument(
        "folder",
        type=Path,
        help="Directory to walk recursively",
    )
    idx_folder.add_argument(
        "--ext",
        default="md,markdown,txt",
        help="Comma-separated extensions without dots (default: md,markdown,txt)",
    )
    idx_folder.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be indexed without calling the embedder",
    )
    idx_folder.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: no limit)",
    )
    idx_folder.set_defaults(_run=_cmd_index_folder)

    ns = p.parse_args(argv)
    if not getattr(ns, "command", None):
        ns.command = "chat"
        ns._run = _cmd_chat
    if ns.command == "ask" and not ns.question:
        p.error("ask requires a question")
    return ns


def _wiki_paths(ns: argparse.Namespace) -> WikiPaths:
    paths = wiki_paths(ns.root)
    ensure_layout(paths)
    return paths


def _cmd_chat(ns: argparse.Namespace) -> None:
    paths = _wiki_paths(ns)
    agent, _k = create_wiki_agent(paths, debug_mode=ns.debug)
    console = Console()
    console.print("[bold]LLM Wiki[/bold] — empty line or bye to exit")
    while True:
        q = Prompt.ask("Wiki", default="").strip()
        if not q or q.lower() in {"bye", "exit", "quit"}:
            break
        agent.print_response(q, markdown=True)


def _cmd_ask(ns: argparse.Namespace) -> None:
    paths = _wiki_paths(ns)
    question = " ".join(ns.question).strip()
    agent, _k = create_wiki_agent(paths, debug_mode=ns.debug)
    agent.print_response(question, markdown=True)


def _cmd_ingest(ns: argparse.Namespace) -> None:
    paths = _wiki_paths(ns)
    agent, _k = create_wiki_agent(paths, debug_mode=ns.debug)
    name = Path(ns.filename).name
    prompt = (
        f"The user added a source under docs/: `{name}`. "
        "Follow the full ingest workflow from your instructions: read the source, "
        "discuss key takeaways with the user before writing if appropriate, "
        "create or update wiki pages (including concept pages and [[wiki-links]]), "
        "update wiki/index.md and append wiki/log.md, then index any new or changed pages."
    )
    agent.print_response(prompt, markdown=True)


def _cmd_reindex(ns: argparse.Namespace) -> None:
    paths = _wiki_paths(ns)
    _a, knowledge = create_wiki_agent(paths, debug_mode=ns.debug)
    indexed = reindex_all_wiki_pages(paths, knowledge)
    console = Console()
    console.print(f"[green]Indexed {len(indexed)} markdown file(s)[/green]")
    for r in indexed:
        console.print(f"  - wiki/{r}")


def _cmd_index_folder(ns: argparse.Namespace) -> None:
    paths = _wiki_paths(ns)
    folder = ns.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    console = Console()
    ext_set = {x.strip() for x in ns.ext.split(",") if x.strip()}
    agent, knowledge = create_wiki_agent(paths, debug_mode=ns.debug)
    agent_with_tool = create_wiki_agent_for_batch_ingestion(paths, debug_mode=ns.debug)
    try:
        indexed = index_folder_into_knowledge(
            knowledge,
            folder,
            ext_set,
            dry_run=ns.dry_run,
            max_files=ns.max_files,
        )
        for file in indexed:
            if "anomaly" in file:
                continue
            full_path = folder / file
            console.print(f"Wiki indexing the file: {full_path} with llm")
            response= agent_with_tool.run(f"Index the file {full_path} and update the wiki index.md following the instructions")
            console.print(response.content)
            user_input = input("Your response:")
            if user_input != "stop":
                response= agent_with_tool.run(user_input)
                console.print(response.content)
            else:
                console.print("Skipping file")
                continue
    except (NotADirectoryError, ValueError) as e:
        raise SystemExit(str(e)) from e

    
    label = "Would index" if ns.dry_run else "Indexed"
    console.print(f"[green]{label} {len(indexed)} file(s) from[/green] {folder}")
    show = indexed[:ns.max_files]
    for r in show:
        console.print(f"  - {r}")
    if len(indexed) > ns.max_files:
        console.print(f"  ... and {len(indexed) - ns.max_files} more")


def main(argv: list[str] | None = None) -> None:
    # Allow running without PYTHONPATH when invoked from llm-wiki directory
    pkg_parent = Path(__file__).resolve().parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    args = _parse_args(argv)
    args._run(args)


if __name__ == "__main__":
    main()
