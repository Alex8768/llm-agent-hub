# LEGACY MODULE — archived persona/experimental logic
# Not used by the LLM-Agent Hub runtime.
# behavior/self_reflector.py
from __future__ import annotations
import asyncio
from datetime import datetime, time as dtime, timedelta
from .insight_logger import collect_reflection

# локальная TZ = системные часы (как ты и просил)
def _seconds_until(target: dtime) -> int:
    now = datetime.now()
    today_target = datetime.combine(now.date(), target)
    if today_target <= now:
        today_target += timedelta(days=1)
    return int((today_target - now).total_seconds())

async def schedule_reflection_daily(at: dtime = dtime(3, 0)):
    """
    Ежедневный запуск ночной рефлексии в указанное локальное время.
    """
    try:
        while True:
            await asyncio.sleep(_seconds_until(at))
            try:
                generated_summary = ""
                summary_text = generated_summary or "Автотест рефлексии: система работает стабильно."
                collect_reflection(summary_text, {"src": "scheduler"})
                print(f"[REFLECT] 🌙 Тестовая ночная запись выполнена: {summary_text}")
            except Exception as e:
                print(f"[REFLECT] Ошибка ночной записи: {e}")
    except asyncio.CancelledError:
        print("[REFLECT] Планировщик остановлен по запросу.")
        raise
