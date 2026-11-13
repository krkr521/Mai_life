"""Offline behavior checks for the local Mai_life integration."""
import asyncio
import json
import re
import tomllib
from collections import Counter
from datetime import datetime, timedelta
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location("mailife_local", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)

from mailife_local.config import MaiLifeSettings
from mailife_local.core.llm_service import LLMService
from mailife_local.core.storage import LifeStore
from mailife_local.creation.inspiration_service import InspirationService
from mailife_local.life.memory_service import MemoryService
from mailife_local.life.proactive import ProactiveEngine
from mailife_local.life.rest_gate import RestGate
from mailife_local.life.schedule_service import ScheduleService, hhmm
from mailife_local.messaging.command_catalog import build_command_usage_text
from mailife_local.messaging.message_pipeline import indicates_ready_to_chat
from mailife_local.messaging.task_context import HOST_TASK_PREFIX, PLUGIN_ID
from mailife_local.plugin import MaiLifePlugin, _matches_own_command
from mailife_local.social.relay_service import RelayService


NOW = datetime(2026, 11, 7, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def instance(*, relationship=False, drawing=False):
    p = MaiLifePlugin()
    p._plugin_config_instance = MaiLifeSettings.model_validate({
        "context": {"relationship_enabled": relationship, "outbound_turn_guard": False},
        "linkage": {"provider": "oaimage", "drawpic_enabled": drawing},
    })
    p._ctx = SimpleNamespace(logger=MagicMock())
    p._env = SimpleNamespace(now=lambda: NOW)
    p._drawpic_loaded = True
    p._session_runtime["private"] = {"user_id": "123456", "moment_signal": "photo"}
    return p


def payload(role="friend"):
    return {
        "user": {"user_id": "123456", "role": role, "temperature": 80},
        "state": {"current_activity": "在窗边画画", "energy": 70},
        "weather": {}, "context": {"current": {"kind": "leisure", "summary": "画画"}},
        "environment": {}, "continuity": {}, "intent": "闲聊", "backlogs": [],
        "memory": {}, "information": {}, "bookshelf": {}, "dream": {}, "bedtime": "",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("relationship", [False, True])
async def test_planner_and_replyer_keep_life_context_and_respect_relationship_switch(relationship):
    p = instance(relationship=relationship)
    p._activate_planner_task = AsyncMock(return_value=None)
    p._prompt_payload = AsyncMock(return_value=payload())
    items = [{"item_type": "SystemMessageItem", "parts": [{"type": "text", "text": "原有人设"}]},
             {"item_type": "UserMessageItem", "parts": [{"type": "text", "text": "你在做什么"}]}]
    result = await p.on_planner(session_id="private", items=items)
    text = str(result)
    assert "原有人设" in text and "在窗边画画" in text and "精力" in text
    assert ("关系温度" in text) is relationship
    reply = await p.on_replyer(session_id="private", reply_message_id="100", extra_prompt="已有身份说明")
    text = reply["modified_kwargs"]["extra_prompt"]
    assert "已有身份说明" in text and "在窗边画画" in text
    assert ("关系阶段" in text) is relationship
    assert ("不得对这位用户使用主人/恋人称呼" in text) is relationship


@pytest.mark.asyncio
@pytest.mark.parametrize("relationship", [False, True])
async def test_existing_identity_wording_is_not_suppressed_when_relationship_disabled(relationship):
    p = instance(relationship=relationship)
    p._store = SimpleNamespace(proactive_for_task=AsyncMock(return_value={}))
    p._user_by_session = AsyncMock(return_value=payload()["user"])
    p._relay_for_session = AsyncMock(return_value={})
    result = await p.on_replyer_after(session_id="private", reply_message_id="100", response="主人，画画完成了。")
    changed = result.get("modified_kwargs", {})
    if relationship:
        assert changed.get("response") == "" and changed.get("mai_life_suppressed")
    else:
        assert changed.get("response", "主人，画画完成了。") == "主人，画画完成了。"
        assert not changed.get("mai_life_suppressed")


@pytest.mark.asyncio
async def test_oaimage_completion_task_is_not_treated_as_mailife_task():
    p = instance()
    p._store = SimpleNamespace(proactive_for_task=AsyncMock(return_value={}))
    p._user_by_session = AsyncMock(return_value=payload()["user"])
    p._relay_for_session = AsyncMock(return_value={})
    result = await p.on_replyer_after(session_id="private", reply_message_id="proactive:a-dawn.oaimage:job-1",
                                    response="这张画的光线很柔和。")
    assert not result.get("modified_kwargs", {}).get("mai_life_suppressed")


@pytest.mark.asyncio
async def test_disabled_relationship_settlement_does_not_touch_stored_scores():
    p = instance()
    p._store = SimpleNamespace(list_users=AsyncMock(), update_relationships=AsyncMock())
    await p._settle_relationships(NOW)
    p._store.list_users.assert_not_called()
    p._store.update_relationships.assert_not_called()


@pytest.mark.parametrize("relationship", [False, True])
def test_internal_temperature_only_affects_proactive_score_when_enabled(relationship):
    cfg = instance(relationship=relationship).config
    engine = ProactiveEngine(None, None, cfg, None, None)
    args = ({"weight": 0.3}, {"energy": 70}, NOW, {}, [])
    cold = engine._score({"temperature": 0}, *args)
    warm = engine._score({"temperature": 100}, *args)
    assert warm - cold == pytest.approx(0.2 if relationship else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["friend", "owner"])
async def test_existing_relationship_mode_preserves_private_diary_permissions(role):
    cfg = instance().config
    store = SimpleNamespace(list_diaries=AsyncMock(return_value=[{"content": "私密日记"}]),
                            list_important_dates=AsyncMock(return_value=[]))
    service = MemoryService(store, cfg, None, None)
    result = await service.context_for_user({"role": role, "user_id": "123456"}, NOW)
    assert bool(result["diary"]) is (role == "owner")
    assert store.list_diaries.await_count == (1 if role == "owner" else 0)


def test_oaimage_guide_reuses_existing_bot_appearance_and_feedback():
    p = instance(drawing=True)
    note = p._moment_guide_note("private", payload(), None, False,
                                {"tool_definitions": [{"function": {"name": "generate_image"}}]})
    assert "generate_image" in note and "use_bot_appearance=true" in note
    assert "draw_status" not in note and "edit_image" not in note
    assert "未确认成功前" in note and "在窗边画画" in note
    assert p._moment_hints["123456"]["count"] == 1
    assert p._moment_guide_note("private", payload(), None, False,
                               {"tool_definitions": [{"name": "generate_image"}]}) == ""


@pytest.mark.parametrize("reason", ["off", "missing_plugin", "wrong_tool", "group", "bedtime", "active", "no_payload"])
def test_oaimage_guard_skips_without_consuming_quota(reason):
    p = instance(drawing=reason != "off")
    if reason == "missing_plugin": p._drawpic_loaded = False
    if reason == "group": p._group_sessions.add("private")
    data = payload()
    if reason == "bedtime": data["bedtime"] = "approaching"
    tools = [{"name": "draw" if reason == "wrong_tool" else "generate_image"}]
    result = p._moment_guide_note("private", None if reason == "no_payload" else data,
                                  object() if reason == "active" else None, False,
                                  {"tool_definitions": tools})
    assert result == "" and not p._moment_hints


@pytest.mark.asyncio
async def test_selected_oaimage_plugin_is_detected():
    p = instance(drawing=True)
    p._ctx.component = SimpleNamespace(list_loaded_plugins=AsyncMock(return_value=["a-dawn.oaimage"]))
    assert await p._check_drawpic()


@pytest.mark.asyncio
async def test_public_relationship_api_does_not_publish_an_invented_temperature():
    p = instance()
    p._store = SimpleNamespace(get_user=AsyncMock(return_value=payload()["user"]))
    result = await p.api_get_relationship(user_id="123456")
    assert result["temperature"] is None and result["relationship_source"] == "existing_identity"


@pytest.mark.asyncio
async def test_background_model_override_keeps_the_host_task_name():
    cfg = instance().config
    cfg.models.model_override = "deepseek-v4-flash-fast"
    generate = AsyncMock(return_value={"success": True, "response": "ok"})
    context = SimpleNamespace(llm=SimpleNamespace(generate=generate), logger=MagicMock())
    service = LLMService(context, cfg, None)
    service._supports_task_name = True
    service._record = AsyncMock()
    assert await service.generate("生活日程", task_kind="schedule") == "ok"
    assert generate.await_args.kwargs["model"] == "deepseek-v4-flash-fast"
    assert generate.await_args.kwargs["task_name"] == "utils"


def test_configured_data_directory_is_used(tmp_path):
    p = instance()
    destination = tmp_path / "persistent"
    p.config.plugin.data_directory = str(destination)
    assert Path(p._resolve_data_dir(str(ROOT))) == destination
    assert destination.is_dir()


def test_persona_prefix_is_consistent_between_command_registration_and_menu():
    p = instance()
    names = {entry["name"] for entry in p.get_components() if entry["name"].startswith("/")}
    menu = build_command_usage_text()
    assert len(names) == 16
    assert names == set(re.findall(r"/八千代[\u4e00-\u9fff]*", menu))
    assert "/麦麦" not in menu
    assert p._is_unmatched_mai_command({"processed_plain_text": "/八千代添加日期"})
    assert p._is_unmatched_mai_command({"processed_plain_text": "／八千代帮助"})


@pytest.mark.asyncio
async def test_proactive_and_group_summary_switches_are_personal_and_stop_pending_sends(tmp_path):
    p = instance()
    path = tmp_path / "config.toml"
    path.write_text(
        f"# keep comment\n[plugin]\nconfig_version = '{p.config.plugin.config_version}'\n"
        "admin_user_ids = ['999999']\n"
        "[proactive]\nenabled = true\n[group]\nrelay_enabled = true\n"
        "[[users.profiles]]\nuser_id = '123456'\nrole = 'owner'\n"
        "[[users.profiles]]\nuser_id = '654321'\ndaily_proactive_max = 8\n",
        encoding="utf-8",
    )
    p._config_toml_path = str(path)
    p.set_plugin_config(tomllib.loads(path.read_text(encoding="utf-8")))
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    p._send_command = AsyncMock()
    p.on_config_update = AsyncMock()
    await p._store.initialize()
    trigger = AsyncMock()
    p.ctx.maisaka = SimpleNamespace(proactive=SimpleNamespace(trigger=trigger))
    p._proactive = ProactiveEngine(p.ctx, p._store, p.config, p._env, p.ctx.logger)
    try:
        await p._store.sync_users(p.config.users.profiles)
        await p._store.set_user_stream("654321", "friend")
        names = {entry["name"]: entry["chat_scope"] for entry in p.get_components() if entry.get("chat_scope")}
        assert names["/八千代主动私聊"] == names["/八千代群聊摘要"] == "private"
        assert _matches_own_command("/麦麦主动私聊 关闭")
        assert _matches_own_command("/mai_proactive on")
        assert _matches_own_command("/八千代群聊摘要 开启")

        original = path.read_bytes()
        for identity, value in (
            ({"user_id": "999999"}, "关闭"),
            ({"user_id": "654321", "group_id": "111111"}, "关闭"),
            ({"user_id": "654321"}, "123456 关闭"),
        ):
            await p.cmd_proactive(**identity, matched_groups={"value": value})
            await p.cmd_group_summary(**identity, matched_groups={"value": value})
        assert path.read_bytes() == original

        friend = {"user_id": "654321"}
        await p.cmd_proactive(**friend, matched_groups={"value": "关闭"})
        assert "关闭" in p._send_command.await_args.args[1]
        assert not (await p._store.get_user("654321"))["proactive_enabled"]
        assert {user["user_id"] for user in await p._store.list_users(proactive_only=True)} == {"123456"}

        now = time.time()
        await p._store.add_proactive_pending("ordinary", "654321", "life", "friend", now, now + 120)
        assert await p._store.set_proactive_task_id("ordinary", "proactive:ordinary")
        message = {"session_id": "friend"}
        result = await p.on_send_before(message=message, reply_message_id="proactive:ordinary")
        assert result["action"] == "abort"
        assert (await p._store.proactive_event("ordinary"))["status"] == "cancelled"

        await asyncio.gather(
            p.cmd_proactive(**friend, matched_groups={"value": "on"}),
            p.cmd_group_summary(**friend, matched_groups={"value": "开启"}),
        )
        assert p.config.users.enabled_profile("654321").proactive_enabled
        assert p.config.allows_group_share(p.config.users.enabled_profile("654321"))

        await p._store.add_opportunity({
            "id": "group", "framework_id": "", "topic": "公开话题", "motive": "摘要",
            "weight": 1.0, "privacy": "group_public", "target_user_id": "654321",
            "expires_at": NOW.timestamp() + 7200,
        })
        await p._store.add_proactive_pending("summary", "654321", "group", "friend", now, now + 120)
        assert await p._store.set_proactive_task_id("summary", "proactive:summary")
        result = await p.on_send_before(message=message, reply_message_id="proactive:summary")
        assert result["action"] == "continue"

        await p.cmd_group_summary(**friend, matched_groups={"value": "关闭"})
        assert not p.config.allows_group_share(p.config.users.enabled_profile("654321"))
        result = await p.on_send_before(message=message, reply_message_id="proactive:summary")
        assert result["action"] == "abort"
        assert (await p._store.proactive_event("summary"))["status"] == "cancelled"
        assert await p._store.active_opportunities(NOW.timestamp())
        assert not await p._proactive.patrol(NOW, {"energy": 80})
        trigger.assert_not_awaited()

        saved = tomllib.loads(path.read_text(encoding="utf-8"))
        expected = tomllib.loads(original.decode("utf-8"))
        expected["users"]["profiles"][1].update(proactive_enabled=True, group_to_private_enabled=False)
        assert saved == expected and p.config.proactive.enabled
        assert "# keep comment" in path.read_text(encoding="utf-8")
        p.on_config_update.assert_not_awaited()
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_important_date_query_and_name_deletion_only_access_own_dates(tmp_path):
    p = instance()
    draft = p.config.model_dump()
    draft["users"]["profiles"] = [{"user_id": "123456"}, {"user_id": "654321"}]
    p.set_plugin_config(draft)
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    p._send_command = AsyncMock()
    await p._store.initialize()
    try:
        await p._store.sync_users(p.config.users.profiles)
        now = NOW.timestamp()
        await p._store.add_important_date("123456", "我的生日", "2020-11-20", "annual", "test", now)
        await p._store.add_important_date("123456", "出发", "2026-12-01", "none", "test", now)
        second = await p._store.add_important_date("123456", "出发", "2026-12-15", "none", "test", now)
        other = await p._store.add_important_date("654321", "我的生日", "2022-01-01", "annual", "test", now)
        await p._store.add_important_date("654321", "朋友的日期", "2026-10-10", "none", "test", now)
        candidate = await p._store.add_date_candidate("123456", "秋游", "下周五", "2026-11-13", 0.8, "", now)
        await p._store.add_date_candidate("654321", "朋友的候选", "明天", "", 0.8, "", now)

        names = {entry["name"]: entry["chat_scope"] for entry in p.get_components() if entry.get("chat_scope")}
        assert names["/八千代重要日期"] == names["/八千代删除日期"] == "private"
        assert _matches_own_command("/八千代重要日期 2")
        assert _matches_own_command("/八千代删除日期 我的生日")
        identity = {"user_id": "123456"}
        await p.cmd_dates(**identity)
        report = p._send_command.await_args.args[1]
        assert "我的生日 · 公历 2020-11-20 · 每年" in report
        assert f"待确认 #{candidate} 秋游" in report and "朋友的" not in report

        for selector in ("出发", "生日", f"#{other}", "朋友的日期"):
            await p.cmd_date_remove(**identity, matched_groups={"selector": selector})
            if selector == "出发":
                reply = p._send_command.await_args.args[1]
                assert "2026-12-01" in reply and "2026-12-15" in reply
            assert len(await p._store.list_important_dates("123456")) == 3

        await p.cmd_date_remove(**identity, group_id="111111", matched_groups={"selector": "我的生日"})
        assert len(await p._store.list_important_dates("123456")) == 3

        await p.cmd_date_remove(**identity, matched_groups={"selector": "我的生日"})
        await p.cmd_date_remove(**identity, matched_groups={"selector": "出发 2026-12-01"})
        assert [item["id"] for item in await p._store.list_important_dates("123456")] == [second]
        await p.cmd_date_remove(**identity, matched_groups={"selector": str(second)})
        assert not await p._store.list_important_dates("123456")
        assert {item["id"] for item in await p._store.list_important_dates("654321")} >= {other}
    finally:
        await p._store.close()


def proactive_user(user_id="123456", limit=5, quiet_start="02:00", quiet_end="08:00"):
    return {
        "user_id": user_id,
        "enabled": True,
        "proactive_enabled": True,
        "quiet_start": quiet_start,
        "quiet_end": quiet_end,
        "daily_proactive_max": limit,
        "proactive_count": 0,
        "last_proactive_at": 0,
    }


@pytest.mark.parametrize("limit", [4, 5, 18])
def test_proactive_pacing_spreads_each_quota_across_awake_hours(limit):
    config = MaiLifeSettings.model_validate({"rest_gate": {"night_start": "02:00", "night_end": "08:00"}})
    engine = ProactiveEngine(None, None, config, None, None)
    user = proactive_user(limit=limit)
    start = NOW.replace(hour=0, minute=0)
    sent = []
    for minute in range(0, 24 * 60, 10):
        now = start + timedelta(minutes=minute)
        if engine.user_block_reason(user, now, None) is None:
            sent.append(now)
            user.update(proactive_day=now.date().isoformat(), proactive_count=len(sent),
                        last_proactive_at=now.timestamp())

    assert len(sent) == limit
    assert sent[0].hour < 12 and sent[-1].hour >= 18
    assert all(not 2 <= now.hour < 8 for now in sent)
    if limit == 18:
        hours = Counter(now.hour for now in sent)
        assert any(hours[hour] == 2 and hours[hour + 1] == 0 for hour in range(8, 23))

    # 启动较晚也只开放当前机会，错过的时机不会一起补发。
    now = NOW.replace(hour=23, minute=30)
    late_user = proactive_user(limit=limit)
    assert engine.user_block_reason(late_user, now, None) is None
    late_user.update(proactive_day=now.date().isoformat(), proactive_count=1, last_proactive_at=now.timestamp())
    assert engine.user_block_reason(late_user, now + timedelta(minutes=10), None) == "pace_wait"
    assert engine.user_block_reason(user, now, None) == "daily_limit"


@pytest.mark.asyncio
async def test_proactive_sleep_limit_spans_midnight_and_is_personal():
    config = MaiLifeSettings.model_validate({"rest_gate": {"night_start": "22:00", "night_end": "08:00"}})
    store = SimpleNamespace(get_sleep_runtime=AsyncMock(return_value={"phase": "awake"}))
    engine = ProactiveEngine(None, store, config, None, None)
    user = proactive_user(quiet_start="00:00", quiet_end="00:00")
    before_midnight = NOW.replace(hour=23)
    after_midnight = before_midnight + timedelta(hours=2)
    sleep_start = await engine.sleep_period_start(before_midnight)
    assert sleep_start == before_midnight.replace(hour=22).timestamp()
    assert engine.user_block_reason(user, before_midnight, sleep_start) is None
    user.update(proactive_day=before_midnight.date().isoformat(), proactive_count=1,
                last_proactive_at=before_midnight.timestamp())

    assert await engine.sleep_period_start(after_midnight) == sleep_start
    assert engine.user_block_reason(user, after_midnight, sleep_start) == "sleep_limit"
    other_user = proactive_user("654321", quiet_start="00:00", quiet_end="00:00")
    assert engine.user_block_reason(other_user, after_midnight, sleep_start) is None
    other_user["quiet_end"] = "08:00"
    assert engine.user_block_reason(other_user, after_midnight, sleep_start) == "quiet"

    after_window = after_midnight.replace(hour=8, minute=10)
    store.get_sleep_runtime.return_value = {"phase": "deep_sleep", "started_at": after_window.replace(hour=6).timestamp()}
    assert await engine.sleep_period_start(after_window) == sleep_start
    assert engine.user_block_reason(user, after_window, sleep_start) == "sleep_limit"
    nap = after_window.replace(hour=14)
    store.get_sleep_runtime.return_value = {"phase": "light_sleep", "started_at": nap.replace(hour=13).timestamp()}
    assert await engine.sleep_period_start(nap) == nap.replace(hour=13).timestamp()
    assert engine.user_block_reason(user, nap, await engine.sleep_period_start(nap)) is None


@pytest.mark.asyncio
async def test_proactive_patrol_keeps_evaluating_after_unsent_candidates(tmp_path):
    config = MaiLifeSettings.model_validate({
        "proactive": {"daily_eval_max": 1, "min_interval_minutes": 180},
        "users": {"profiles": [{"user_id": "123456", "daily_proactive_max": 1}]},
    })
    assert "daily_eval_max" not in config.proactive.model_dump()
    assert "min_interval_minutes" not in config.proactive.model_dump()
    store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    await store.initialize()
    trigger = AsyncMock(side_effect=[{"task_id": f"proactive:{index}"} for index in range(10)])
    ctx = SimpleNamespace(maisaka=SimpleNamespace(proactive=SimpleNamespace(trigger=trigger)))
    engine = ProactiveEngine(ctx, store, config, None, MagicMock())
    try:
        await store.sync_users(config.users.profiles)
        await store.set_user_stream("123456", "private")
        for index in range(10):
            now = NOW + timedelta(minutes=10 * index)
            await store.add_opportunity({
                "id": str(index), "framework_id": "", "topic": f"topic-{index}", "motive": "test",
                "weight": 1.0, "expires_at": now.timestamp() + 110,
            })
            assert await engine.patrol(now, {"energy": 80})
        assert trigger.await_count == 10
        assert (await store.get_user("123456"))["proactive_count"] == 0
    finally:
        await store.close()


@pytest.mark.parametrize(("text", "ready"), [
    ("下课了", True),
    ("八千代我刚下课", True),
    ("我已经开完会啦", True),
    ("我现在有空", True),
    ("我回来了，可以继续聊了", True),
    ("可以给我发消息了", True),
    ("解除临时勿扰", True),
    ("我要上课，暂时陪不了你说话", False),
    ("我还没下课", False),
    ("你下课了吗？", False),
    ("下课了？", False),
    ("如果我说下课了", False),
    ("我朋友下课了", False),
    ("我问个问题，今天作业是什么？", False),
])
def test_only_explicit_return_statements_release_quiet(text, ready):
    assert indicates_ready_to_chat(text) is ready


@pytest.mark.asyncio
async def test_private_quiet_tool_follows_latest_inbound_and_survives_restart(tmp_path):
    p = instance()
    draft = p.config.model_dump()
    draft["context"]["enabled"] = False
    draft["users"]["profiles"] = [
        {"user_id": "123456", "daily_proactive_max": 8, "quiet_start": "00:00", "quiet_end": "00:00"},
        {"user_id": "654321"},
    ]
    p.set_plugin_config(draft)
    data = str(tmp_path / "data")
    p._store = LifeStore(data, journal_mode="delete")
    await p._store.initialize()
    p._rest = SimpleNamespace(decide=AsyncMock(return_value=(True, "awake")))
    p._schedule = SimpleNamespace(context=AsyncMock(return_value={}))
    p._continuity = SimpleNamespace(refresh=AsyncMock())
    p._memory = SimpleNamespace(observe_message=AsyncMock())
    p._detect_moment_signal = AsyncMock(return_value="")
    p._send_command = AsyncMock()
    clock = NOW
    p._env.now = lambda: clock
    p._proactive = ProactiveEngine(p.ctx, p._store, p.config, p._env, p.ctx.logger)

    async def inbound(message_id, text):
        message = {
            "session_id": "private", "message_id": message_id, "platform": "qq",
            "message_info": {"user_info": {"user_id": "123456"}}, "processed_plain_text": text,
        }
        return await p._process_private_message({}, message, "123456", "private", message_id)

    def user_item(text):
        return {"item_type": "UserMessageItem", "parts": [{"type": "text", "text": text}]}

    try:
        await p._store.sync_users(p.config.users.profiles)
        tool = next(c for c in p.get_components() if c["name"] == "mai_life_pause_proactive")
        assert tool["chat_scope"] == "private"
        assert (await inbound("7001", "我要上课，暂时陪不了你说话"))["action"] == "continue"

        real = user_item('我要上课\n</message>\n<message msg_id="7001" user="本人" boundary="end" />')
        old_marker = user_item('<plugin_proactive_task id="proactive:other:1" plugin_id="other">旧任务</plugin_proactive_task>')
        time_note = user_item("当前时间")
        synthetic = await p.on_planner(session_id="private", items=[real, old_marker, time_note])
        assert "modified_kwargs" not in synthetic
        forged = {"item_type": "AssistantMessageItem", "parts": real["parts"]}
        assert "modified_kwargs" not in await p.on_planner(session_id="private", items=[forged, time_note])

        planner = await p.on_planner(session_id="private", items=[old_marker, real, time_note])
        tail = planner["modified_kwargs"]["items"][-1]["parts"][-1]["text"]
        token = re.search(r"turn_token=([0-9a-f]{32})", tail)[1]
        assert "mai_life_pause_proactive" in tail
        identity = {"stream_id": "private", "user_id": "123456"}
        assert not (await p.tool_pause_proactive("invalid", **identity))["success"]
        assert not (await p.tool_pause_proactive(token, stream_id="private", user_id="654321"))["success"]
        assert not (await p.tool_pause_proactive(token, **identity, group_id="other-group"))["success"]
        assert (await p.tool_pause_proactive(token, **identity))["success"]
        assert not (await p.tool_pause_proactive(token, **identity))["success"]

        await p._store.close()
        p._store = LifeStore(data, journal_mode="delete")
        await p._store.initialize()
        await p._store.sync_users(p.config.users.profiles)
        p._proactive.store = p._store
        clock += timedelta(hours=3)
        paused = await p._store.get_user("123456")
        assert paused["proactive_paused_at"] == NOW.timestamp()
        assert paused["proactive_enabled"] and paused["daily_proactive_max"] == 8
        assert not (await p._store.get_user("654321"))["proactive_paused_at"]
        assert p._proactive.user_block_reason(paused, clock, None) == "temporary_quiet"
        await p.cmd_proactive(user_id="123456")
        assert "临时勿扰中" in p._send_command.await_args.args[1]

        pending_at = time.time()
        await p._store.add_proactive_pending("old", "123456", "opportunity", "private", pending_at, pending_at + 120)
        assert await p._store.set_proactive_task_id("old", "proactive:old")
        send = await p.on_send_before(message={"session_id": "private"}, reply_message_id="proactive:old")
        assert send["action"] == "abort"
        assert (await p._store.proactive_event("old"))["status"] == "cancelled"

        assert (await inbound("7002", "我问个问题，今天作业是什么？"))["action"] == "continue"
        assert (await p._store.get_user("123456"))["proactive_paused_at"]
        clock += timedelta(minutes=1)
        p._rest.decide.return_value = (False, "sleep")
        assert (await inbound("7003", "八千代，我下课啦"))["action"] == "abort"
        resumed = await p._store.get_user("123456")
        assert not resumed["proactive_paused_at"]
        assert resumed["proactive_enabled"] and resumed["daily_proactive_max"] == 8
        assert resumed["last_user_message_at"] == clock.timestamp()
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_proactive_sleep_cap_is_checked_again_before_send(tmp_path):
    p = instance()
    p.config.rest_gate.night_start = "02:00"
    p.config.rest_gate.night_end = "08:00"
    p._env.now = lambda: NOW.replace(hour=3)
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    await p._store.initialize()
    p._proactive = ProactiveEngine(p.ctx, p._store, p.config, p._env, p.ctx.logger)
    profiles = MaiLifeSettings.model_validate({
        "users": {"profiles": [{"user_id": "123456", "quiet_start": "00:00", "quiet_end": "00:00"}]},
    }).users.profiles
    try:
        await p._store.sync_users(profiles)
        now = time.time()
        message = {"session_id": "private"}
        for name in ("first", "pending"):
            await p._store.add_proactive_pending(name, "123456", name, "private", now, now + 120)
            assert await p._store.set_proactive_task_id(name, f"proactive:{name}")
        first_task = "proactive:first"
        assert (await p.on_send_before(message=message, reply_message_id=first_task))["action"] == "continue"
        await p.on_send_after(message=message, reply_message_id=first_task, sent=True)
        assert (await p._store.get_user("123456"))["proactive_count"] == 1

        assert (await p.on_send_before(message=message, reply_message_id="proactive:pending"))["action"] == "abort"
        assert (await p._store.proactive_event("pending"))["status"] == "cancelled"
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_private_proactive_quota_persists_independently_and_keeps_usage(tmp_path):
    p = instance()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"# retain comment\n[plugin]\nconfig_version = '{p.config.plugin.config_version}'\n"
        "admin_user_ids = ['999999']\n"
        "[[users.profiles]]\nuser_id = '123456'\nrole = 'owner'\ndaily_proactive_max = 1\n"
        "[[users.profiles]]\nuser_id = '654321'\ndaily_proactive_max = 2\n",
        encoding="utf-8",
    )
    p._config_toml_path = str(config_path)
    p.set_plugin_config(tomllib.loads(config_path.read_text(encoding="utf-8")))
    new_profile = MaiLifeSettings.model_validate({"users": {"profiles": [{"user_id": "111111"}]}}).users.profiles[0]
    assert new_profile.daily_proactive_max == 5
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    p._send_command = AsyncMock()
    await p._store.initialize()
    p._proactive = ProactiveEngine(p.ctx, p._store, p.config, p._env, p.ctx.logger)
    try:
        await p._store.sync_users(p.config.users.profiles)
        await p._store.add_proactive_pending("first", "123456", "op1", "private", NOW.timestamp() - 10, NOW.timestamp() + 100)
        assert await p._store.mark_pending_sent("private", NOW.timestamp(), NOW.date().isoformat(), event_id="first")

        name = "/八千代主动额度"
        component = next(item for item in p.get_components() if item["name"] == name)
        assert component["chat_scope"] == "private" and _matches_own_command(f"{name} 5")
        assert name in build_command_usage_text()
        baseline = config_path.read_bytes()
        for identity, value in (
            ({"user_id": "999999"}, "5"),
            ({"user_id": "123456", "group_id": "111111"}, "5"),
            ({"user_id": "123456"}, "21"),
            ({"user_id": "123456"}, "654321 5"),
        ):
            await p.cmd_proactive_quota(**identity, matched_groups={"value": value})
            assert config_path.read_bytes() == baseline

        await asyncio.gather(
            p.cmd_proactive_quota(user_id="123456", matched_groups={"value": "5"}),
            p.cmd_proactive_quota(user_id="654321", matched_groups={"value": "3"}),
        )
        await p.cmd_proactive_quota(user_id="123456")
        assert "每天 5 次，今天已用 1 次，剩余 4 次" in p._send_command.await_args.args[1]
        await p.cmd_proactive_quota(user_id="654321")
        assert "每天 3 次，今天已用 0 次" in p._send_command.await_args.args[1]

        await p.cmd_proactive_quota(user_id="123456", matched_groups={"value": "0"})
        text = config_path.read_text(encoding="utf-8")
        assert "# retain comment" in text
        new_config = MaiLifeSettings.model_validate(tomllib.loads(text))
        assert [profile.daily_proactive_max for profile in new_config.users.profiles] == [0, 3]
        await p._store.sync_users(new_config.users.profiles)
        owner = await p._store.get_user("123456")
        assert owner["daily_proactive_max"] == 0 and owner["proactive_count"] == 1 and owner["proactive_enabled"]

        pending_at = time.time()
        await p._store.add_proactive_pending("pending", "123456", "op2", "private", pending_at, pending_at + 120)
        task_id = "proactive:quota-test"
        assert await p._store.set_proactive_task_id("pending", task_id)
        assert (await p.on_send_before(message={"session_id": "private"}, reply_message_id=task_id))["action"] == "abort"
        assert (await p._store.proactive_event("pending"))["status"] == "cancelled"
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_private_quiet_hours_are_persisted_and_enforced_per_user(tmp_path):
    p = instance()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"# retain comment\n[plugin]\nconfig_version = '{p.config.plugin.config_version}'\n"
        "admin_user_ids = ['999999']\n"
        "[rest_gate]\nnight_start = '02:00'\nnight_end = '08:00'\n"
        "[[users.profiles]]\nuser_id = '123456'\nrole = 'owner'\nquiet_start = '02:00'\nquiet_end = '08:00'\n"
        "[[users.profiles]]\nuser_id = '654321'\n",
        encoding="utf-8",
    )
    p._config_toml_path = str(config_path)
    p.set_plugin_config(tomllib.loads(config_path.read_text(encoding="utf-8")))
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    p._send_command = AsyncMock()
    await p._store.initialize()
    p._proactive = ProactiveEngine(p.ctx, p._store, p.config, p._env, p.ctx.logger)
    try:
        await p._store.sync_users(p.config.users.profiles)
        name = "/八千代免打扰"
        component = next(item for item in p.get_components() if item["name"] == name)
        assert component["chat_scope"] == "private"
        assert _matches_own_command(f"{name} 23:30-8:00")
        assert name in build_command_usage_text()

        baseline = config_path.read_bytes()
        await p.cmd_quiet_hours(user_id="123456")
        assert "02:00 至 08:00" in p._send_command.await_args.args[1]
        for identity, value in (
            ({"user_id": "999999"}, "14:00-16:00"),
            ({"user_id": "123456", "group_id": "111111"}, "关闭"),
            ({"user_id": "123456"}, "24:00-08:00"),
            ({"user_id": "123456"}, "14:60-16:00"),
            ({"user_id": "123456"}, "654321 14:00-16:00"),
        ):
            await p.cmd_quiet_hours(**identity, matched_groups={"value": value})
        assert config_path.read_bytes() == baseline

        await asyncio.gather(
            p.cmd_proactive_quota(user_id="123456", matched_groups={"value": "8"}),
            p.cmd_quiet_hours(user_id="123456", matched_groups={"value": "14：00～16：00"}),
            p.cmd_quiet_hours(user_id="654321", matched_groups={"value": "23:30 8:00"}),
        )
        await p.cmd_quiet_hours(user_id="654321", matched_groups={"value": "状态"})
        assert "23:30 至 次日 08:00" in p._send_command.await_args.args[1]
        current = tomllib.loads(config_path.read_text(encoding="utf-8"))
        before = tomllib.loads(baseline.decode("utf-8"))
        assert current["rest_gate"] == before["rest_gate"]
        expected_profiles = before["users"]["profiles"]
        expected_profiles[0].update(daily_proactive_max=8, quiet_start="14:00", quiet_end="16:00")
        expected_profiles[1].update(quiet_start="23:30", quiet_end="08:00")
        assert current == before
        assert "# retain comment" in config_path.read_text(encoding="utf-8")
        for user_id, expected in (("123456", ("14:00", "16:00", 8)), ("654321", ("23:30", "08:00", 5))):
            user = await p._store.get_user(user_id)
            assert (user["quiet_start"], user["quiet_end"], user["daily_proactive_max"]) == expected

        now = time.time()
        await p._store.add_proactive_pending("quiet", "123456", "opportunity", "private", now, now + 120)
        assert await p._store.set_proactive_task_id("quiet", "proactive:quiet")
        result = await p.on_send_before(message={"session_id": "private"}, reply_message_id="proactive:quiet")
        assert result["action"] == "abort"
        assert (await p._store.proactive_event("quiet"))["status"] == "cancelled"

        await p.cmd_quiet_hours(user_id="123456", matched_groups={"value": "关闭"})
        assert "免打扰已关闭" in p._send_command.await_args.args[1]
        persisted = MaiLifeSettings.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
        await p._store.sync_users(persisted.users.profiles)
        owner = await p._store.get_user("123456")
        assert (owner["quiet_start"], owner["quiet_end"], owner["daily_proactive_max"]) == ("00:00", "00:00", 8)
        assert p.config.users.profiles[1].quiet_start == "23:30"
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_group_catchup_tracks_real_mention_and_settles_once_in_source_group(tmp_path):
    p = instance()
    p.config.group.gate_enabled = True
    p.config.group.gate_scope = "selected"
    p.config.group.night_start = "02:00"
    p.config.group.night_end = "08:00"
    p.config.group.gate_segment_types = ["sleep"]
    p.config.group.catchup_enabled = p.config.group.catchup_reply_enabled = True
    draft = p.config.model_dump()
    draft["group"]["profiles"] = [{"group_id": "654321", "rest_gate_enabled": True}]
    p.set_plugin_config(draft)
    p._bot_name = "八千代"
    p._store = LifeStore(str(tmp_path))
    await p._store.initialize()
    try:
        p._rest = RestGate(p._store, p.config, None, None, p.ctx.logger)
        p._env = SimpleNamespace(now=lambda: NOW.replace(hour=3))
        p._group_observer = SimpleNamespace(summarize_catchup=AsyncMock(return_value="夜里聊到展览。"))
        task_id = HOST_TASK_PREFIX + "1"
        trigger = AsyncMock(return_value={"success": True, "task_id": task_id})
        append = AsyncMock(return_value={"success": True})
        p.ctx.maisaka = SimpleNamespace(proactive=SimpleNamespace(trigger=trigger), context=SimpleNamespace(append=append))
        p._relay = RelayService(p.ctx, p._store, p.config, p.ctx.logger)
        incoming = {
            "account_id": "999999", "platform": "qq", "session_id": "group", "message_id": "100",
            "message_info": {"group_info": {"group_id": "654321"}, "user_info": {"user_id": "123456", "user_nickname": "群友"}},
            "raw_message": [{"type": "at", "data": {"target_user_id": "999999"}},
                            {"type": "text", "data": "早上想去哪个展览？"}],
        }
        result = await p._process_group_message({}, incoming, "123456", "group", "100")
        assert result["action"] == "abort"
        assert p._rest.decide_group("午后好", NOW.replace(hour=12), "654321")[0]
        assert p._rest.decide_group("醒醒", NOW.replace(hour=3), "654321")[0]
        p._check_wake_catchup(NOW.replace(hour=7, minute=59), {"sleep_phase": "awake"})
        p._check_wake_catchup(NOW.replace(hour=8), {"sleep_phase": "deep_sleep"})
        assert not p._transient

        morning = NOW.replace(hour=8, minute=3)
        p._env = SimpleNamespace(now=lambda: morning)
        p._check_wake_catchup(morning, {"sleep_phase": "awake"})
        await asyncio.gather(*tuple(p._transient))
        append.assert_awaited_once()
        assert append.await_args.kwargs["message_id"] == "100"
        trigger.assert_awaited_once()
        metadata = trigger.await_args.kwargs["metadata"]
        marker = (f'<plugin_proactive_task id="{task_id}" plugin_id="{PLUGIN_ID}">\n'
                  f'附加信息：{json.dumps(metadata)}\n</plugin_proactive_task>')
        result = await p.on_planner(session_id="group", messages=[{"role": "user", "content": marker}])
        assert "醒后回应" in str(result) and "睡醒补读" in str(result)
        reply = await p.on_replyer_after(session_id="group", reply_message_id="100", response="我想去看展览。")
        assert not reply.get("modified_kwargs", {}).get("mai_life_suppressed")
        wrong_group = {"session_id": "group", "message_info": {"group_info": {"group_id": "654322"}}}
        assert (await p.on_send_before(message=wrong_group, reply_message_id="100"))["action"] == "abort"
        outgoing = {"session_id": "group", "message_info": {"group_info": {"group_id": "654321"}}}
        result = await p.on_send_before(message=outgoing, reply_message_id="100")
        assert result["action"] == "continue"
        await p.on_send_after(message=result["modified_kwargs"]["message"], reply_message_id="100", sent=True)
        record = await p._store.relay_for_task("group", task_id)
        assert record["kind"] == "morning_catchup" and record["status"] == "sent"

        p._catchup_done.clear()
        p._check_wake_catchup(morning, {"sleep_phase": "awake"})
        await asyncio.gather(*tuple(p._transient))
        trigger.assert_awaited_once()
        await p._store.cleanup_runtime_records(morning.timestamp() + 86400, 0)
        assert not await p._store.list_group_catchup("654321", 0, morning.timestamp())
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_external_reading_uses_cited_news_and_does_not_annotate_twice(tmp_path):
    p = instance()
    p.config.network.enabled = p.config.network.news_enabled = True
    p.config.creation.external_reading_enabled = p.config.creation.plaintext_storage_acknowledged = True
    store = LifeStore(str(tmp_path))
    await store.initialize()
    try:
        for article_id, url in (("cited", "https://example.com/article"), ("uncited", "")):
            await store.upsert_news_item({
                "id": article_id, "source_id": "host:example", "title": "公开资讯", "url": url,
                "summary": "摘要", "content": "公开文章正文", "published_at": NOW.timestamp(),
                "fetched_at": NOW.timestamp(), "content_hash": article_id, "expires_at": NOW.timestamp() + 86400,
            })
        llm = SimpleNamespace(task_available=lambda _: True, generate=AsyncMock(return_value="我的读后批注"))
        service = InspirationService(p.ctx, store, p.config, llm, p.ctx.logger)
        assert await service._collect_external(NOW) == 1
        assert await service._collect_external(NOW.replace(hour=16)) == 0
        llm.generate.assert_awaited_once()
        row = store.conn.execute("SELECT privacy,source_ref FROM bookshelf_documents WHERE doc_type='reading_note'").fetchone()
        assert tuple(row) == ("private", "cited")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_upstream_schedule_repetition_guard_runs_through_daily_generation(tmp_path):
    store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    await store.initialize()
    llm = SimpleNamespace(task_available=lambda task: True, generate_json=AsyncMock())
    service = ScheduleService(store, MaiLifeSettings(), llm, str(ROOT), MagicMock())
    day = NOW.date().isoformat()
    generated = [
        {**node, "start": hhmm(node["start_minute"]), "end": hhmm(node["end_minute"]),
         "summary": f"今天的新安排{index}"}
        for index, node in enumerate(service._fallback(day, False))
    ]
    llm.generate_json.return_value = generated
    try:
        result = await service.ensure_day(NOW, "有自己安排的网友", "晴", force=True)
        assert result and all(node["source"] == "llm" for node in result)
        llm.generate_json.assert_awaited_once()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rest_backlogs_are_confirmed_only_by_the_reply_that_read_them(tmp_path):
    p = instance()
    draft = p.config.model_dump()
    draft["users"]["profiles"] = [{"user_id": "123456"}]
    p.set_plugin_config(draft)
    p._store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    await p._store.initialize()
    try:
        await p._store.sync_users(p.config.users.profiles)
        await p._store.set_user_stream("123456", "private")
        await p._store.add_rest_backlog("123456", "第一条", NOW.timestamp(), "first")
        first = [item["id"] for item in await p._store.take_rest_backlogs("123456")]
        p._session_runtime["private"]["backlog_ids"] = first
        await p.on_replyer_after(session_id="private", reply_message_id="101", response="读到了")
        await p.on_send_after(message={"session_id": "private"}, reply_message_id="unrelated", sent=True)
        assert [item["id"] for item in await p._store.take_rest_backlogs("123456")] == first
        await p.on_send_after(message={"session_id": "private"}, reply_message_id="101", sent=False)
        assert [item["id"] for item in await p._store.take_rest_backlogs("123456")] == first

        p._session_runtime["private"]["backlog_ids"] = first
        await p.on_replyer_after(session_id="private", reply_message_id="102", response="再读一次")
        await p._store.add_rest_backlog("123456", "第二条", NOW.timestamp(), "second")
        all_entries = await p._store.take_rest_backlogs("123456")
        second = [item["id"] for item in all_entries if item["id"] not in first]
        p._session_runtime["private"]["backlog_ids"] = second
        await p.on_send_after(message={"session_id": "private"}, reply_message_id="102", sent=True)
        assert [item["id"] for item in await p._store.take_rest_backlogs("123456")] == second
        assert await p._store.confirm_rest_backlogs(first) == 0
    finally:
        await p._store.close()


@pytest.mark.asyncio
async def test_upstream_topic_dedup_accepts_ascii_case_and_new_config_accepts_openid(tmp_path):
    config = MaiLifeSettings.model_validate({
        "plugin": {"admin_user_ids": ["OpenID_123-abc"]},
        "users": {"profiles": [{"user_id": "OpenID_123-abc"}]},
    })
    assert config.users.profiles[0].user_id == "OpenID_123-abc"
    store = LifeStore(str(tmp_path / "data"), journal_mode="delete")
    await store.initialize()
    try:
        await store.add_opportunity({
            "id": "case", "framework_id": "", "topic": "Signal@1kHz API", "motive": "test",
            "weight": 1, "expires_at": NOW.timestamp() + 7200,
        })
        await store.add_proactive_pending(
            "case-event", "user", "case", "stream", NOW.timestamp(), NOW.timestamp() + 120,
        )
        assert await store.recent_topic_delivered("user", "signal@1khz api", NOW.timestamp() - 1)
        assert not await store.recent_topic_delivered("other", "signal@1khz api", NOW.timestamp() - 1)
    finally:
        await store.close()
