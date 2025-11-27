# cognition/file_watcher.py
"""
Watch the data/ folder, embedding and storing changes via rag_ingest.embed_and_store.
"""

from pathlib import Path
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .rag_ingest import DATA_DIR, embed_and_store, EXCLUDE_FILES


class DataDirHandler(FileSystemEventHandler):
    def _handle(self, path: Path):
        name = path.name

        # Ignore hidden/temp files or those blocked by policy
        if name.startswith(".") or name.startswith("~") or name in EXCLUDE_FILES:
            print(f"[WATCH] ↩ Ignored: {name}")
            return

        try:
            embed_and_store(path)
        except Exception as e:
            print(f"[WATCH] ❌ Indexing error for {name}: {e}")
            traceback.print_exc()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        print(f"[WATCH] 🆕 New file: {path.name}")
        self._handle(path)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        print(f"[WATCH] ✏️ File updated: {path.name}")
        self._handle(path)


def start_data_watch():
    """Start background watcher; invoked from main.on_startup()."""
    observer = Observer()
    handler = DataDirHandler()
    observer.schedule(handler, str(DATA_DIR), recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[WATCH] 👀 Watching data directory: {DATA_DIR}")
