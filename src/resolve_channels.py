"""Run with python -m src.resolve_channels --help."""
import argparse
import csv
import json
import logging
import sqlite3
import sys
from pathlib import Path
from .database import DEFAULT_DB, ROOT, connect, confirm, normalize, now
from .youtube_client import APIError, YouTubeClient


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def show(items, fetched_at):
    configure_console()
    print(f"\nAPI snapshot: {fetched_at}")
    for item in items:
        row = normalize(item)
        print("\n" + "\n".join(f"{k}: {v if v is not None else 'unavailable'}"
              for k, v in row.items() if k != "raw_json"))
        print(f"Channel URL: https://www.youtube.com/channel/{item['id']}")
    if not items:
        print("No available channel matches on this page.")


def search(db, seeds, pages, refresh=False):
    client = None
    failures = 0
    for seed in seeds:
        token = ""
        print(f"\n=== {seed} ===")
        for _ in range(pages):
            cached = db.execute("SELECT * FROM search_pages WHERE seed_query=? AND page_token=?", (seed, token)).fetchone()
            if cached and not refresh:
                items, next_token, fetched = json.loads(cached["items_json"]), cached["next_page_token"], cached["fetched_at"]
            else:
                try:
                    client = client or YouTubeClient()
                    items, next_token = client.search_page(seed, token)
                except APIError as exc:
                    logging.error("Seed %r: %s", seed, exc)
                    failures += 1
                    if exc.fatal:
                        return 1
                    break
                fetched = now()
                with db:
                    if refresh and token == "":
                        db.execute("DELETE FROM search_pages WHERE seed_query=?", (seed,))
                    db.execute("INSERT OR REPLACE INTO search_pages VALUES (?,?,?,?,?)",
                               (seed, token, next_token, json.dumps(items, ensure_ascii=False), fetched))
            show(items, fetched)
            if not next_token:
                break
            token = next_token
        else:
            if token:
                print("More results exist; increase --pages to fetch additional pages.")
    return 1 if failures else 0


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="Phase 1: discover, review and confirm YouTube channels.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("search", help="Fetch/cache five candidates per seed per page")
    discover.add_argument("--seeds", type=Path, default=ROOT / "seeds.txt")
    discover.add_argument("--query", action="append", help="Seed query; repeat for multiple names")
    discover.add_argument("--pages", type=int, default=1)
    discover.add_argument("--refresh", action="store_true")
    review = sub.add_parser("review", help="Review cached candidates without API calls")
    review.add_argument("--seed")
    save = sub.add_parser("confirm", help="Explicitly confirm a previously reviewed candidate")
    save.add_argument("--seed", required=True)
    save.add_argument("--channel-id", required=True)
    sub.add_parser("list", help="List confirmed channels")
    export = sub.add_parser("export", help="Export confirmed channels as UTF-8 CSV")
    export.add_argument("--output", type=Path, default=ROOT / "data" / "exports" / "channels.csv")
    args = parser.parse_args(argv)
    if args.command == "search" and args.pages < 1:
        parser.error("--pages must be positive")
    (ROOT / "data").mkdir(exist_ok=True)
    logger = logging.getLogger()
    if not any(getattr(handler, "resolver_log", False) for handler in logger.handlers):
        handler = logging.FileHandler(ROOT / "data" / "resolver.log", encoding="utf-8")
        handler.resolver_log = True
        logger.addHandler(handler)
        if not any(type(h) is logging.StreamHandler for h in logger.handlers):
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.WARNING)
    db = connect(args.db)
    try:
        if args.command == "search":
            seeds = args.query or args.seeds.read_text(encoding="utf-8-sig").splitlines()
            seeds = list(dict.fromkeys(s.strip() for s in seeds if s.strip() and not s.lstrip().startswith("#")))
            if not seeds:
                parser.error("No nonempty seed queries provided")
            return search(db, seeds, args.pages, args.refresh)
        if args.command in ("review", "confirm"):
            rows = db.execute("SELECT * FROM search_pages WHERE (? IS NULL OR seed_query=?) ORDER BY fetched_at DESC",
                              (args.seed, args.seed)).fetchall()
            for row in rows:
                items = json.loads(row["items_json"])
                if args.command == "review":
                    print(f"\n=== {row['seed_query']} ===")
                    show(items, row["fetched_at"])
                else:
                    for item in items:
                        if item["id"] == args.channel_id:
                            confirm(db, item, args.seed, row["fetched_at"])
                            print(f"Confirmed {item['snippet']['title']} ({item['id']}) in {args.db}")
                            return 0
            if args.command == "confirm":
                parser.error("Channel ID was not found among cached candidates for this exact seed")
            if not rows:
                print("No cached candidates. Run search first.")
        elif args.command == "list":
            rows = db.execute("SELECT channel_id,channel_title,custom_url FROM channels ORDER BY channel_title").fetchall()
            for row in rows:
                print(" | ".join(str(v or "") for v in row))
            print(f"{len(rows)} confirmed channel(s).")
        elif args.command == "export":
            if args.output.resolve() == args.db.resolve():
                parser.error("Export output must not overwrite the database")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cursor = db.execute("SELECT * FROM channels ORDER BY channel_id")
            with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([c[0] for c in cursor.description])
                writer.writerows(cursor)
            print(f"Exported confirmed channels to {args.output}")
        return 0
    except (OSError, sqlite3.Error) as exc:
        logging.error("Local storage failure (%s). Check paths and permissions.", type(exc).__name__)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; committed pages and confirmations are saved.")
        raise SystemExit(130)
