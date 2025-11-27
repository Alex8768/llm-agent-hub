import os, sys, io, time

# ----- configuration -----
EXCLUDE_DIRS = {'.git', '.venv', '__pycache__', 'data', '_backups', '_chatpack',
                'node_modules', '.idea', '.vscode', '.DS_Store'}
INCLUDE_EXT = {'.py', '.json', '.md', '.txt', '.sh', '.yml', '.yaml',
               '.toml', '.ini', '.html', '.css', '.js', '.ts', '.tsx', '.sql'}
EXCLUDE_GLOBS = ('.db', '.db3', '.sqlite', '.sqlite3', '.zip', '.png', '.jpg',
                 '.jpeg', '.pdf', '.mp4', '.mov', '.bin', '.pth')
MAX_FILE_BYTES = 1_000_000         # upper size bound for a single file
MAX_TOTAL_BYTES = 19_000_000       # total size cap (~19 MB)
HEADER_TMPL = "\n===== {path} =====\n"

def is_texty_file(path):
    name = os.path.basename(path).lower()
    if any(name.endswith(g) for g in EXCLUDE_GLOBS):
        return False
    _, ext = os.path.splitext(name)
    return ext in INCLUDE_EXT

def should_skip_dir(name):
    return name in EXCLUDE_DIRS or name.startswith('.DS')

def iter_files(root):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # in-place directory filter
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            yield rel, full

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'all_code_20mb.txt'
    manifest_path = 'CODE_MANIFEST.txt'

    total = 0
    included = []
    skipped = []

    # open output file in text mode (UTF-8)
    with io.open(out_path, 'w', encoding='utf-8', errors='ignore') as out:
        for rel, full in iter_files(root):
            if not is_texty_file(full):
                skipped.append((rel, 'ext_excluded'))
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                skipped.append((rel, 'stat_error'))
                continue
            if size > MAX_FILE_BYTES:
                skipped.append((rel, f'file_too_big({size})'))
                continue

            header = HEADER_TMPL.format(path=rel)
            budget_left = MAX_TOTAL_BYTES - total
            need_bytes = len(header.encode('utf-8'))
            if budget_left <= need_bytes:
                skipped.append((rel, 'no_budget_for_header'))
                break

            # read content
            try:
                with io.open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                skipped.append((rel, 'read_error'))
                continue

            data = header + content
            data_bytes = len(data.encode('utf-8'))
            if total + data_bytes <= MAX_TOTAL_BYTES:
                out.write(data)
                total += data_bytes
                included.append(rel)
            else:
                # if it does not fit, try a truncated copy
                room = MAX_TOTAL_BYTES - total - need_bytes
                if room > 0:
                    out.write(HEADER_TMPL.format(path=rel))
                    # slice by characters (already unicode)
                    trimmed = content[:room]  # bytes!=chars, but errors='ignore' will guard
                    out.write(trimmed)
                    total = MAX_TOTAL_BYTES
                    included.append(rel + ' [TRUNCATED]')
                else:
                    skipped.append((rel, 'no_budget_left'))
                break

    # write manifest
    with io.open(manifest_path, 'w', encoding='utf-8') as mf:
        mf.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        mf.write(f"Root: {os.path.abspath(root)}\n")
        mf.write(f"Output: {os.path.abspath(out_path)}\n")
        mf.write(f"TotalBytes<= {MAX_TOTAL_BYTES}\n\n")
        mf.write("=== INCLUDED ===\n")
        for p in included:
            mf.write(p + "\n")
        mf.write("\n=== SKIPPED ===\n")
        for p, reason in skipped:
            mf.write(f"{p} :: {reason}\n")

    print(f"Done. Wrote {out_path} (<= {MAX_TOTAL_BYTES} bytes).")
    print(f"Manifest: {manifest_path}")
    print(f"Included: {len(included)} files; Skipped: {len(skipped)} files.")

if __name__ == '__main__':
    main()
