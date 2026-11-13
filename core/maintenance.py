"""Independent maintenance jobs: keep strict failures visible and retry them later."""
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import asyncio


class MaintenanceJobs:
    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.jobs: dict[str, dict[str, Any]] = {}

    async def run(self, name: str, now: datetime, action: Callable[[], Awaitable[Any]],
                  timeout: float, *, force: bool = False) -> tuple[bool, Any]:
        day = now.date().isoformat()
        previous = self.jobs.get(name, {})
        if previous.get("day") != day:
            previous = {}
        if not force and float(previous.get("next_retry_at", 0)) > now.timestamp():
            return False, None
        job = {"day": day, "status": "running", "failures": int(previous.get("failures", 0)),
               "error": "", "next_retry_at": 0.0}
        self.jobs[name] = job
        try:
            result = await asyncio.wait_for(action(), timeout=timeout)
        except asyncio.CancelledError:
            job["status"] = "pending"
            raise
        except Exception as exc:
            job["status"] = "failed"
            job["failures"] += 1
            job["error"] = f"{type(exc).__name__}: {exc}"[:500]
            delay = min(1800, 600 * job["failures"])
            job["next_retry_at"] = now.timestamp() + delay
            self.logger.error(f"[MaiLife] 维护任务 {name} 失败，{delay//60} 分钟后重试: {job['error']}")
            return False, None
        job.update(status="ready", failures=0)
        return True, result

    def snapshot(self, now: datetime) -> dict[str, dict[str, Any]]:
        day = now.date().isoformat()
        return {name: dict(job) for name, job in self.jobs.items() if job.get("day") == day}

    def summary(self, now: datetime, *, has_schedule: bool) -> str:
        jobs = self.snapshot(now)
        failed = [f"{name}: {job['error']}（{datetime.fromtimestamp(job['next_retry_at'], now.tzinfo).strftime('%H:%M')} 重试）"
                  for name, job in jobs.items() if job["status"] == "failed"]
        if failed:
            return "；".join(failed)
        if not has_schedule:
            return "今日日程尚未通过审核，等待生成；生活状态暂停推进"
        return "正常"


def current_life_state(state: dict[str, Any], context: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Do not present yesterday's simulated scene as the current day's fact."""
    updated = float(state.get("last_updated_at") or 0)
    same_day = bool(updated) and datetime.fromtimestamp(updated, now.tzinfo).date() == now.date()
    if context.get("nodes") and same_day:
        return state
    return {**state, "current_activity": "当前活动未确认，不推测正在做的事", "current_location": "未知",
            "sleep_phase": "unknown", "health_note": "生活状态尚未更新，不作当前事实描述"}
