"""Cross-plugin regression checks using actual APIs and isolated SQLite stores."""
from datetime import datetime, timedelta
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
def load(name, path, *, package=False):
    spec = util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

life = load("unified_life", ROOT / "__init__.py", package=True)
provider_root = ROOT.parent / ("affinity" if ROOT.name == "life" else "yufei-pan_maibot-impression-card-plugin")
identity_root = ROOT.parent / ("identity" if ROOT.name == "life" else "Alcpare_maibot_identity_plugin")
affinity = load("unified_affinity", provider_root / "plugin.py", package=True)
identity = load("unified_identity", identity_root / "plugin.py")
from unified_life.config import MaiLifeSettings
from unified_life.core.storage import LifeStore
from unified_life.core.relationship_bridge import RelationshipBridge, relationship_fields, relationship_note, relationship_weight
from unified_life.creation.bookshelf_service import BookshelfService
from unified_life.life.proactive import ProactiveEngine
from unified_life.plugin import MaiLifePlugin

NOW = datetime(2026, 11, 13, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
UID = "100000001"


async def stack(tmp_path, *, total=8.0, owner=True):
    scores = affinity.AffinityPlugin()
    scores.set_plugin_config(affinity.AffinityPluginConfig().model_dump())
    scores._store = affinity.AffinityStore(tmp_path / "affinity.db")
    scores._store.outreach_policy = affinity.OutreachPolicy(enabled=True)
    await scores._store.configure_relationships(
        affinity.RelationshipPolicy(protected_user_ids=frozenset({UID})),
    )
    await scores._store.upsert(affinity.AffinityRecord(
        person_id="person-owner", platform="qq", user_id=UID, total=total,
        scores={"trust": 7.5, "familiarity": 8.0}, display_name="测试用户",
    ))
    labels = identity.IdentityPlugin()
    config = identity.IdentityPluginConfig().model_dump()
    config["identity"]["roster"] = UID + " = 主人" if owner else ""
    config["dynamic_relationship"]["enabled"] = True
    labels.set_plugin_config(config)
    labels._rebuild_roster()

    async def score_call(name, **args):
        if name == "com.0-hz.impression-card.get_relationship":
            value = await scores.relationship_api(**args)
        elif name == "com.0-hz.impression-card.apply_outreach_feedback":
            value = await scores.outreach_feedback_api(**args)
        else:
            raise AssertionError(name)
        return {"success": True, "result": value}
    labels._ctx = SimpleNamespace(api=SimpleNamespace(call=score_call), logger=MagicMock())

    async def identity_call(name, **args):
        if name == "Alcpare.identity.get_relationship":
            return {"success": True, "result": await labels.relationship_api(**args)}
        return await score_call(name, **args)

    bot = MaiLifePlugin()
    config = MaiLifeSettings().model_dump()
    config["context"].update(relationship_provider="existing_identity", relationship_enabled=False,
                             outreach_feedback_enabled=True, outbound_turn_guard=False)
    # Intentionally disagree with the manual roster: local roles and temperatures must not authorize anyone.
    config["users"]["profiles"] = [{"user_id": UID, "role": "friend", "initial_temperature": 77}]
    bot.set_plugin_config(config)
    bot._ctx = SimpleNamespace(api=SimpleNamespace(call=identity_call), logger=MagicMock())
    bot._store = LifeStore(str(tmp_path / "life"), journal_mode="delete")
    await bot._store.initialize()
    bot._configure_relationship_bridge()
    await bot._store.sync_users(bot.config.users.profiles)
    await bot._store.set_user_stream(UID, "private")
    bot._env = SimpleNamespace(now=lambda: NOW)
    return bot, labels, scores


async def close(bot, scores):
    await bot._store.close()
    scores._store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("total", [-4.5, 8, 25.3])
async def test_real_api_chain_preserves_raw_affinity_and_manual_owner(tmp_path, total):
    bot, labels, scores = await stack(tmp_path, total=total)
    try:
        result = await bot.api_get_relationship(user_id=UID)
        assert result["role"] == "owner" and result["role_source"] == "manual"
        assert result["affinity"] == total and result["temperature"] is None
        assert result["dimensions"]["trust"] == 7.5
        user = await bot._store.get_user(UID)
        assert BookshelfService.allow_private(user)
        assert "核实为主人" in relationship_note(user)
        assert (await bot._store.management_overview_counts())["owners"] == 1
        assert await bot._store.first_interaction_day(UID) == ""
        raw = bot._store.conn.execute("SELECT role,temperature FROM users").fetchone()
        assert tuple(raw) == ("friend", 77)
        await bot._settle_relationships(NOW)
        assert bot._store.conn.execute("SELECT temperature FROM users").fetchone()[0] == 77
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_proactive_planner_and_replyer_receive_verified_owner_with_internal_system_disabled(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        data = {"state": {}, "weather": {}, "context": {}, "user": await bot._store.get_user(UID),
                "dream": {}, "backlogs": [], "environment": {}, "continuity": {}, "intent": "",
                "memory": {}, "information": {}, "bookshelf": {}, "bedtime": ""}
        bot._prompt_payload = AsyncMock(return_value=data)
        bot._activate_planner_task = AsyncMock(return_value=None)
        items = [{"item_type": "SystemMessageItem", "parts": [{"type": "text", "text": "原有人设"}]},
                 {"item_type": "UserMessageItem", "parts": [{"type": "text", "text": "主动分享私人作品"}]}]
        planner = await bot.on_planner(session_id="private", items=items)
        text = json.dumps(planner, ensure_ascii=False)
        assert "核实为主人" in text and "现有好感 8" in text and "原有人设" in text
        assert "关系温度" not in text
        replyer = await bot.on_replyer(session_id="private", reply_message_id="101", extra_prompt="原有身份说明")
        assert "核实为主人" in replyer["modified_kwargs"]["extra_prompt"]
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_missing_profile_and_provider_failure_never_reuse_old_owner_or_temperature(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        with scores._store._connect() as db:
            db.execute("DELETE FROM affinity")
        user = await bot._store.get_user(UID)
        assert user["role"] == "owner" and relationship_fields(user, False)["affinity"] is None
        assert "分值未知" in relationship_note(user)
        async def unavailable(*args, **kwargs):
            raise RuntimeError("provider offline")
        bot._ctx.api.call = unavailable
        user = await bot._store.get_user(UID)
        assert user["role"] == "unknown" and user["temperature"] is None
        assert not BookshelfService.allow_private(user)
        assert not relationship_fields(user, False)["relationship_ready"]
        assert "provider offline" in relationship_fields(user, False)["relationship_error"]
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_high_affinity_cannot_create_owner_and_manual_revocation_applies_immediately(tmp_path):
    bot, labels, scores = await stack(tmp_path, total=1000)
    try:
        assert (await bot._store.get_user(UID))["role"] == "owner"
        labels.config.identity.roster = ""
        labels._rebuild_roster()
        user = await bot._store.get_user(UID)
        assert user["role"] == "friend" and not BookshelfService.allow_private(user)
        assert relationship_weight(user, False) == 0.2
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_owner_only_pending_send_is_aborted_after_manual_owner_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[MaiLifePlugin.__module__].time, "time", lambda: NOW.timestamp())
    bot, labels, scores = await stack(tmp_path)
    try:
        ts = NOW.timestamp()
        await bot._store.add_opportunity({"id": "private-work", "framework_id": "", "topic": "私人作品", "motive": "分享",
                                         "privacy": "owner_only", "weight": 1, "expires_at": ts + 3600})
        await bot._store.add_proactive_pending("event", UID, "private-work", "private", ts, ts + 120)
        await bot._store.set_proactive_task_id("event", "proactive:private")
        bot._proactive = ProactiveEngine(bot.ctx, bot._store, bot.config, bot._env, bot.ctx.logger)
        labels.config.identity.roster = ""
        labels._rebuild_roster()
        result = await bot.on_send_before(message={"session_id": "private"}, reply_message_id="proactive:private")
        assert result["action"] == "abort"
        assert any("owner_permission_unconfirmed" in str(call) for call in bot.ctx.logger.info.call_args_list)
        assert (await bot._store.proactive_event("event"))["status"] == "cancelled"
    finally:
        await close(bot, scores)


async def delivered(store, day, *, count=1, eligible=True):
    at = NOW + timedelta(days=day)
    for index in range(count):
        event = f"event-{day}-{index}"
        await store.add_proactive_pending(event, UID, "op", "private", at.timestamp(), at.timestamp() + 120)
        assert await store.mark_pending_sent("private", at.timestamp(), at.date().isoformat(),
                                             event_id=event, response_eligible=eligible)
    return at


@pytest.mark.asyncio
async def test_observations_wait_48_hours_and_ignore_undelivered_and_sleep_messages(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        await bot._store.add_proactive_pending("failed", UID, "op", "private", NOW.timestamp(), NOW.timestamp() + 120)
        await delivered(bot._store, 0, count=12)
        await delivered(bot._store, 1, eligible=False)
        assert bot._store.conn.execute("SELECT COUNT(*) FROM outreach_observations").fetchone()[0] == 1
        assert not await bot._store.outreach_feedback_due(NOW.timestamp() + 48*3600 - 1)
        due = await bot._store.outreach_feedback_due(NOW.timestamp() + 48*3600)
        assert len(due) == 1 and due[0]["outcome"] == "unanswered"
        assert (await bot._store.proactive_event("failed"))["sent_at"] == 0
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_delayed_ordinary_chat_counts_as_response_and_later_return_recovers(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        await delivered(bot._store, 0)
        at = NOW + timedelta(hours=24)
        await bot._store.record_interaction(UID, "今天又来聊一会", at.timestamp(), at.hour, "real-user")
        due = await bot._store.outreach_feedback_due(NOW.timestamp() + 49*3600)
        assert due[0]["outcome"] == "answered"
        await bot._relationship_bridge.settle_feedback(bot._store, NOW + timedelta(days=4))
        before = (await scores._store.get("person-owner")).total
        await bot._relationship_bridge.settle_feedback(bot._store, NOW + timedelta(days=4))
        assert (await scores._store.get("person-owner")).total == before
        assert not await bot._store.outreach_feedback_due((NOW + timedelta(days=4)).timestamp())
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pause", "quiet", "disabled", "sync"])
async def test_known_unavailability_excludes_open_observation_days(tmp_path, monkeypatch, mode):
    bot, labels, scores = await stack(tmp_path)
    try:
        await delivered(bot._store, 0)
        at = NOW + timedelta(minutes=5)
        if mode == "pause":
            await bot._store.record_interaction(UID, "我要上课了", at.timestamp(), at.hour, "busy")
            assert await bot._store.pause_proactive_for_message(UID, "private", at.timestamp())
        else:
            monkeypatch.setattr(sys.modules[LifeStore.__module__].time, "time", lambda: at.timestamp())
            if mode=="sync":
                bot.config.users.profiles[0].proactive_enabled=False
                await bot._store.sync_users(bot.config.users.profiles)
            else:
                args = {"quiet_start": "09:00", "quiet_end": "17:00"} if mode == "quiet" else {"proactive_enabled": False}
                await bot._store.set_user_proactive_preferences(UID, **args)
        due = await bot._store.outreach_feedback_due(NOW.timestamp() + 49*3600)
        assert due and due[0]["outcome"] == "excused"
        await bot._relationship_bridge.settle_feedback(bot._store, NOW + timedelta(days=3))
        assert (await scores._store.get("person-owner")).total == 8
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_small_bounded_effects_recovery_idempotency_and_trust_is_unchanged(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        start = NOW.date()
        results = []
        for day in range(45):
            results.append(await scores._store.apply_outreach("qq", UID, str(start+timedelta(days=day)), "unanswered"))
        assert [item["delta"] for item in results[:3]] == [0, 0, -0.02]
        record = await scores._store.get("person-owner")
        assert record.total == pytest.approx(7.4)
        assert record.scores["trust"] == 7.5 and record.scores["familiarity"] == 8
        assert (await bot._store.get_user(UID))["role"] == "owner"
        day = str(start+timedelta(days=45))
        result = await scores._store.apply_outreach("qq", UID, day, "returned")
        assert result["delta"] == 0.04 and result["missed_days"] == 0
        repeated = await scores._store.apply_outreach("qq", UID, day, "returned")
        assert repeated["duplicate"]
        with pytest.raises(ValueError, match="不同结果"):
            await scores._store.apply_outreach("qq", UID, day, "unanswered")
        assert (await scores._store.get("person-owner")).total == pytest.approx(7.44)
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_feedback_missing_profile_has_no_synthetic_score_and_rolls_back_atomically(tmp_path, monkeypatch):
    bot, labels, scores = await stack(tmp_path)
    try:
        assert (await scores._store.apply_outreach("qq", "999999999", str(NOW.date()), "answered"))["missing"]
        assert (await scores._store.get_relationship("qq", "999999999"))["affinity"]["total"] is None
        def fail(*args, **kwargs):
            raise RuntimeError("derived relation failure")
        monkeypatch.setattr(affinity, "update_relationship", fail)
        with pytest.raises(RuntimeError, match="derived relation failure"):
            await scores._store.apply_outreach("qq", UID, str(NOW.date()), "answered")
        assert (await scores._store.get("person-owner")).total == 8
        assert scores._store._connect().execute("SELECT COUNT(*) FROM outreach_affinity_feedback").fetchone()[0] == 0
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
async def test_model_write_cannot_overwrite_concurrent_outreach_settlement(tmp_path):
    bot, labels, scores = await stack(tmp_path)
    try:
        stale = await scores._store.get("person-owner")
        for day in range(3):
            await scores._store.apply_outreach("qq", UID, str(NOW.date()+timedelta(days=day)), "unanswered")
        stale.total += 0.1
        stale.scores["responsiveness"] = 100
        await scores._store.upsert(stale)
        final = await scores._store.get("person-owner")
        assert final.total == pytest.approx(8.08)
        assert final.scores["responsiveness"] == pytest.approx(4.9)
    finally:
        await close(bot, scores)


@pytest.mark.asyncio
@pytest.mark.parametrize("sleeping", [False, True])
async def test_platform_confirmation_collects_only_awake_eligible_deliveries(tmp_path, sleeping):
    bot, labels, scores = await stack(tmp_path)
    try:
        ts=NOW.timestamp()
        await bot._store.add_opportunity({"id":"op","framework_id":"","topic":"日常分享","motive":"闲聊",
                                         "weight":1,"privacy":"normal","expires_at":ts+3600})
        await bot._store.add_proactive_pending("event",UID,"op","private",ts,ts+120)
        await bot._store.set_proactive_task_id("event","proactive:test")
        active=SimpleNamespace(task_id="proactive:test",record_id="event",reply_anchor="101",sent=False)
        bot._active_tasks=SimpleNamespace(current=AsyncMock(return_value=active),
                                          reserve_send=AsyncMock(return_value=True),finish_send=AsyncMock())
        bot._proactive=SimpleNamespace(sleep_period_start=AsyncMock(return_value=ts if sleeping else None),
                                       user_block_reason=MagicMock(return_value=None))
        before=await bot.on_send_before(message={"session_id":"private"},reply_message_id="101",set_reply=False)
        assert before["action"]=="continue"
        message=before["modified_kwargs"]["message"]
        assert message["message_info"]["additional_config"]["mai_life_outreach_eligible"] is (not sleeping)
        await bot.on_send_after(message=message,reply_message_id="101",sent=True)
        assert bot._store.conn.execute("SELECT COUNT(*) FROM outreach_observations").fetchone()[0]==int(not sleeping)
    finally:
        await close(bot,scores)


@pytest.mark.asyncio
async def test_positive_effect_has_a_cap_and_out_of_order_writes_are_rejected(tmp_path):
    bot,labels,scores=await stack(tmp_path)
    try:
        for offset in range(40):
            await scores._store.apply_outreach("qq",UID,str(NOW.date()+timedelta(days=offset)),"answered")
        assert (await scores._store.get("person-owner")).total==pytest.approx(8.3)
        with pytest.raises(ValueError,match="日期顺序"):
            await scores._store.apply_outreach("qq",UID,str(NOW.date()-timedelta(days=1)),"unanswered")
        user=await bot._store.get_user(UID)
        profile=bot.config.users.enabled_profile(UID)
        profile.group_to_private_enabled=True
        bot.config.group.relay_enabled=True
        bot.config.group.owner_group_to_private_enabled=False
        assert not bot.config.allows_group_share(profile,user["role"])
    finally:
        await close(bot,scores)


@pytest.mark.asyncio
async def test_private_background_cache_changes_with_authoritative_permissions(tmp_path):
    bot,labels,scores=await stack(tmp_path)
    try:
        bot.config.context.background_cache_minutes=10
        bot._payload_cache=type(bot._payload_cache)(600)
        bot._env.snapshot=lambda *args,**kwargs:{}
        bot._schedule=SimpleNamespace(context=AsyncMock(return_value={}))
        bot._bedtime=SimpleNamespace(approaching=lambda now:False)
        for attr in ("_llm","_state","_rest","_proactive","_continuity","_group_observer","_relay","_creation","_admin"):
            setattr(bot,attr,object())
        async def memory(user,now):
            return {"diary":{"content":"private-diary"} if user["role"]=="owner" else {}}
        async def shelf(user):
            return {"items":[{"title":"private-book","privacy":"private"}] if user["role"]=="owner" else []}
        bot._memory=SimpleNamespace(context_for_user=AsyncMock(side_effect=memory))
        bot._bookshelf=SimpleNamespace(context_for_user=AsyncMock(side_effect=shelf))
        bot._information=SimpleNamespace(context=AsyncMock(return_value={}))
        original=await bot._prompt_payload("private")
        assert original["bookshelf"]["items"]
        await bot._prompt_payload("private")
        assert bot._bookshelf.context_for_user.await_count==1
        labels.config.identity.roster=""
        labels._rebuild_roster()
        revoked=await bot._prompt_payload("private")
        assert not revoked["bookshelf"]["items"] and not revoked["memory"]["diary"]
        assert bot._bookshelf.context_for_user.await_count==2
    finally:
        await close(bot,scores)
