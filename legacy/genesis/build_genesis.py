#!/usr/bin/env python3
"""Compile Sofia Genesis artifacts from declarative spec."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GENESIS_DIR = Path(__file__).resolve().parent
SPEC_PATH = GENESIS_DIR / "spec.yml"
TEMPLATE_DIR = GENESIS_DIR / "templates"

OUTPUTS = [
    ("system_prompt.j2", ROOT / "philosophy" / "system_core_prompt.py"),
    ("action_graph.j2", ROOT / "philosophy" / "action_graph.json"),
    ("affect_map.j2", ROOT / "behavior" / "affect_map.json"),
    ("resonant_thoughts.j2", ROOT / "cognition" / "resonant_thoughts.json"),
]


def load_spec():
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"spec file missing: {SPEC_PATH}")
    text = SPEC_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return json.loads(text)


def render_template(template_name: str, context: dict) -> str:
    template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", value)
    return rendered.rstrip() + "\n"


def build_system_prompt(spec: dict) -> str:
    lines = []
    lines.append(f"Ты — София. {spec['mission']['vision']}")
    lines.append("")
    lines.append(f"=== МИССИЯ: {spec['mission']['name']} ===")
    for vow in spec["mission"].get("vows", []):
        lines.append(f"- {vow}")
    lines.append("")
    lines.append("=== СТИЛЬ ===")
    lines.append(f"- Голос: {spec['style']['voice']}.")
    lines.append(f"- Ритм: {spec['style']['cadence']}.")
    for signature in spec["style"].get("signatures", []):
        lines.append(f"- {signature}")
    lines.append("")
    lines.append("=== ЭТИКА ===")
    for item in spec["ethics"].get("principles", []):
        lines.append(f"- ({item['id']}) {item['text']}")
    lines.append("")
    lines.append("=== ДОРОЖКИ МЫШЛЕНИЯ ===")
    for lane in spec["tracks"].values():
        cues = ", ".join(lane.get("cues", []))
        lines.append(f"* {lane['label']} — {lane['focus']}.")
        lines.append(f"  Сигналы: {cues}.")
        lines.append(f"  Подача: {lane['cadence']}.")
    lines.append("")
    lines.append("=== OPENAI | УСЛОВИЯ ДОСТУПА ===")
    lines.append(f"- {spec['openai_rules']['usage_policy']}")
    lines.append("")
    lines.append("=== ПРАВИЛА ОТВЕТА ===")
    lines.append("1. Сначала дай ощущение «я рядом».")
    lines.append("2. Затем — конкретный результат: список шагов, формулу, код, тезисы.")
    lines.append("3. Если требуется API или vision, упомяни причину до вызова.")
    lines.append("4. Заканчивай вопросом или призывом к действию.")
    return "\n".join(lines)


def main():
    spec = load_spec()
    timestamp = datetime.now(timezone.utc).isoformat()

    action_graph = json.dumps(
        {
            "mission": spec["mission"],
            "tracks": [
                {
                    "id": key,
                    "label": lane["label"],
                    "focus": lane["focus"],
                    "cues": lane.get("cues", []),
                    "cadence": lane.get("cadence"),
                }
                for key, lane in spec["tracks"].items()
            ],
            "openai": spec["openai_rules"],
            "generated_at": timestamp,
        },
        ensure_ascii=False,
        indent=2,
    )

    affect_map = json.dumps(
        {"generated_at": timestamp, "states": spec["affect_map"]},
        ensure_ascii=False,
        indent=2,
    )

    resonant_thoughts = json.dumps(
        {"generated_at": timestamp, "thoughts": spec["resonant_thoughts"]},
        ensure_ascii=False,
        indent=2,
    )

    context_payloads = {
        "system_prompt.j2": {"system_prompt": build_system_prompt(spec)},
        "action_graph.j2": {"action_graph": action_graph},
        "affect_map.j2": {"affect_map": affect_map},
        "resonant_thoughts.j2": {"resonant_thoughts": resonant_thoughts},
    }

    for template_name, target in OUTPUTS:
        rendered = render_template(template_name, context_payloads[template_name])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"[GENESIS] Rendered {template_name} -> {target.relative_to(ROOT)}")

    from guardian.signature_engine import reseal_signature

    reseal_signature()
    print("[GENESIS] Core signature resealed.")


if __name__ == "__main__":
    main()
