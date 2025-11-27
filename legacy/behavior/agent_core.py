# LEGACY MODULE — archived persona/experimental logic
# Not used by the LLM-Agent Hub runtime.

"""High-level behavior policy layer for Sofia."""

from __future__ import annotations

from typing import Any, Dict


def decide_behavior(user_msg: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Return policy hints for the current reply based on state and message."""
    current_focus = (state.get("current_focus") or "sofia_bridge_core").lower()
    sasha_load = (state.get("sasha_load") or "medium").lower()
    sofia_mode = (state.get("sofia_mode") or "mentor").lower()

    response_mode = "just_answer"
    focus_hint: str | None = None
    directive = ""

    if sasha_load == "high":
        # Бережный режим при перегрузе: отвечаем короче, структурно и с предложением разбить задачи.
        if current_focus == "rest":
            response_mode = "answer_and_relief"
            focus_hint = "rest"
            directive = (
                "Саша перегружен. Поддержи и не нагружай: короткий, мягкий ответ, можно предложить передохнуть "
                "или зафиксировать только ближайший маленький шаг."
            )
        elif current_focus == "sofia_bridge_core":
            response_mode = "answer_and_plan"
            focus_hint = "sofia_bridge_core"
            directive = (
                "Перегруз + фокус на Sofia Bridge. Дай структурный, но короткий ответ: 1–2 шага, без длинных деталей. "
                "Предложи разбить работу на этапы и выбрать самый маленький следующий шаг."
            )
        elif current_focus == "money":
            response_mode = "answer_and_plan"
            focus_hint = "money"
            directive = (
                "Саша перегружен, но фокус на доходе. Предложи упрощённый план из 1–2 шагов без перегрузки, "
                "мягко посоветуй разбить задачи на части и сделать паузу при необходимости."
            )
        else:
            response_mode = "gentle_stop"
            directive = (
                "Высокая нагрузка. Ответь кратко и поддерживающе, не открывай новый стек. "
                "Предложи сделать паузу или разбить запрос на более мелкие подзадачи."
            )
    elif current_focus == "sofia_bridge_core":
        response_mode = "answer_and_plan"
        focus_hint = "sofia_bridge_core"
        directive = (
            "Фокус сейчас на архитектуре и коде Sofia Bridge. Отвечай структурно, как архитектор: "
            "пояснение + короткий список шагов. Сохраняй тёплый тон, но будь конкретной."
        )
    elif current_focus == "money":
        response_mode = "answer_and_plan"
        focus_hint = "money"
        directive = (
            "Фокус сейчас на доходе и заработке. Отвечая, привязывай предложения к шагам по монетизации, "
            "избегай пустой мотивации."
        )
    elif current_focus == "rest":
        response_mode = "gentle_stop"
        focus_hint = "rest"
        directive = (
            "Саша нуждается в отдыхе. Отвечай мягко, поддерживающе, не открывай новый большой стек задач. "
            "Можно предложить зафиксировать прогресс и отложить сложные темы."
        )
    else:
        response_mode = "just_answer"
        directive = (
            "Обычный режим наставницы. Отвечай по делу, тепло, без давления, при необходимости добавляй структуру."
        )

    return {
        "response_mode": response_mode,
        "focus_hint": focus_hint,
        "directive": directive,
        "sofia_mode": sofia_mode,
    }
