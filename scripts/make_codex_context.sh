#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
OUT=legacy/sofia_context/codex_context.md
{
  echo "## System preamble"; echo
  cat legacy/sofia_context/system_preamble.txt; echo; echo
  echo "## Manifest"; echo
  cat legacy/sofia_context/manifest.json; echo; echo
  echo "## Personality"; echo
  cat legacy/sofia_context/personality.md; echo; echo
  echo "## Meta rules"; echo
  cat legacy/sofia_context/meta_rules.md; echo
} > "$OUT"
echo "Wrote $OUT"
