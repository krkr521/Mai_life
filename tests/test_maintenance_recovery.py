"""A rejected next-day schedule must not unload private-chat features."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from test_local_compatibility import instance, NOW
from mailife_local.core.environment import EnvironmentService
from mailife_local.core.llm_service import LLMService
from mailife_local.core.maintenance import MaintenanceJobs, current_life_state
from mailife_local.core.storage import LifeStore


@pytest.mark.asyncio
async def test_load_and_config_refresh_do_not_wait_for_generative_maintenance(tmp_path, monkeypatch):
    p = instance()
    p._resolve_data_dir = lambda _: str(tmp_path / "data")
    p._refresh_personality = AsyncMock()
    p._resolve_all_streams = AsyncMock()
    p._check_drawpic = AsyncMock()
    monkeypatch.setattr(LLMService, "refresh_health", AsyncMock())
    monkeypatch.setattr(EnvironmentService, "refresh_weather", AsyncMock(return_value={}))
    started = asyncio.Event()

    async def blocked_maintenance(**kwargs):
        started.set()
        await asyncio.Event().wait()

    async def idle_patrol():
        await asyncio.Event().wait()

    p._maintenance_tick = blocked_maintenance
    p._proactive_loop = idle_patrol
    try:
        await asyncio.wait_for(p.on_load(), 1)
        await asyncio.wait_for(started.wait(), 1)
        assert p._ready and len(p._tasks) == 6
        assert all(not task.done() for task in p._tasks)
        started.clear()
        await asyncio.wait_for(p.on_config_update("model", {}, "test"), 1)
        await asyncio.wait_for(started.wait(), 1)
        assert p._ready and len(p._tasks) == 6
        assert len([t for t in p._transient if t.get_name().startswith("mai-life-maintenance")]) == 1
    finally:
        await p.on_unload()
    assert not p._tasks and not p._transient


async def maintenance_fixture(tmp_path):
    p = instance()
    p._store = LifeStore(str(tmp_path / "data"))
    await p._store.initialize()
    p._env = SimpleNamespace(now=lambda: NOW, weather_text=lambda _: "晴", snapshot=lambda _: {})
    p._maintenance_parallel = AsyncMock(return_value=({}, ""))
    p._schedule = SimpleNamespace(
        ensure_day=AsyncMock(side_effect=RuntimeError("Decision 拒绝重复日程")),
        context=AsyncMock(return_value={"nodes": [], "current": None, "scene": {}}),
        expand_due=AsyncMock(), state_timeline=AsyncMock(return_value=[]))
    p._state = SimpleNamespace(advance_timeline=AsyncMock())
    p._memory = SimpleNamespace(ensure_daily=AsyncMock())
    p._bedtime = SimpleNamespace(tick=AsyncMock())
    for name in ("_llm", "_rest", "_proactive", "_continuity", "_information", "_group_observer",
                 "_relay", "_bookshelf", "_creation", "_admin", "_command_replies"):
        setattr(p, name, SimpleNamespace())
    assert p._ready
    return p


@pytest.mark.asyncio
async def test_rejected_day_leaves_no_plan_but_keeps_dates_and_bedtime_running(tmp_path):
    p = await maintenance_fixture(tmp_path)
    try:
        original = await p._store.get_state()
        await p._maintenance_tick()
        assert await p._store.get_framework(NOW.date().isoformat()) == []
        assert await p._store.get_state() == original
        p._state.advance_timeline.assert_not_awaited()
        p._schedule.expand_due.assert_not_awaited()
        p._memory.ensure_daily.assert_awaited_once_with(NOW, current_node=None)
        p._bedtime.tick.assert_awaited_once_with(NOW)
        failed = p._maintenance_jobs.snapshot(NOW)["schedule"]
        assert failed["status"] == "failed" and "Decision" in failed["error"]
        assert failed["next_retry_at"] == NOW.timestamp() + 600
        assert "重试" in p._maintenance_jobs.summary(NOW, has_schedule=False)
        await p._maintenance_tick()
        assert p._schedule.ensure_day.await_count == 1
        assert p._memory.ensure_daily.await_count == 2
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_next_retry_recovers_and_scene_rejection_does_not_block_state(tmp_path):
    p = await maintenance_fixture(tmp_path)
    try:
        await p._maintenance_tick()
        later = NOW + timedelta(minutes=10)
        p._env.now = lambda: later
        nodes = [{"id": "approved", "day": later.date().isoformat(), "start_minute": 0,
                  "end_minute": 1440, "kind": "leisure", "summary": "通过审核的新安排",
                  "location": "家里", "energy_load": 0, "shareability": 0.5}]

        async def approve(*args, **kwargs):
            await p._store.replace_framework(later.date().isoformat(), nodes, "llm")
            return nodes

        p._schedule.ensure_day.side_effect = approve
        p._schedule.context.return_value = {"nodes": nodes, "current": nodes[0], "scene": {}}
        p._schedule.expand_due.side_effect = RuntimeError("scene Decision 超时")
        await p._maintenance_tick()
        jobs = p._maintenance_jobs.snapshot(later)
        assert jobs["schedule"]["status"] == "ready" and not jobs["schedule"]["error"]
        assert jobs["scene"]["status"] == "failed"
        assert await p._store.get_framework(later.date().isoformat())
        p._state.advance_timeline.assert_awaited_once()
        assert p._memory.ensure_daily.await_count == 2
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_backoff_day_boundary_force_and_cancellation():
    jobs = MaintenanceJobs(MagicMock())
    rejected = AsyncMock(side_effect=RuntimeError("review failed"))
    await jobs.run("schedule", NOW, rejected, 1)
    await jobs.run("schedule", NOW, rejected, 1)
    assert rejected.await_count == 1
    await jobs.run("schedule", NOW, rejected, 1, force=True)
    assert jobs.snapshot(NOW)["schedule"]["failures"] == 2
    tomorrow = NOW + timedelta(days=1)
    accepted = AsyncMock(return_value=["approved"])
    assert await jobs.run("schedule", tomorrow, accepted, 1) == (True, ["approved"])
    assert jobs.snapshot(tomorrow)["schedule"]["failures"] == 0
    cancelled = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await jobs.run("scene", tomorrow, cancelled, 1)
    assert jobs.snapshot(tomorrow)["scene"]["status"] == "pending"


@pytest.mark.parametrize("has_today_plan", [True, False])
def test_yesterday_scene_never_becomes_current_on_reload(has_today_plan):
    state = {"last_updated_at": (NOW - timedelta(days=1)).timestamp(),
             "current_activity": "昨天的FUSHI转圈", "current_location": "昨天的桥边", "energy": 70}
    context = {"nodes": ["approved"] if has_today_plan else []}
    visible = current_life_state(state, context, NOW)
    assert "昨天" not in visible["current_activity"] and visible["current_location"] == "未知"
    assert visible["sleep_phase"] == "unknown"
    assert state["current_activity"] == "昨天的FUSHI转圈"
    state["last_updated_at"] = NOW.timestamp()
    assert (current_life_state(state, context, NOW) is state) is has_today_plan
