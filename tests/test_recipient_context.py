"""Recipient-scoped proactive context reaches both models without changing send state."""
from unittest.mock import AsyncMock

import json
import time

import pytest

from test_local_compatibility import instance
from mailife_local.config import MaiLifeSettings
from mailife_local.core.storage import LifeStore
from mailife_local.messaging.task_context import ActivePluginTask


async def prepare(tmp_path):
    plugin=instance()
    plugin._plugin_config_instance=MaiLifeSettings.model_validate({
        "context":{"enabled":False},
        "users":{"profiles":[{"user_id":"123456"},{"user_id":"654321"}]},
    })
    plugin._store=LifeStore(str(tmp_path/"data"))
    await plugin._store.initialize()
    await plugin._store.sync_users(plugin.config.users.profiles)
    now=time.time()
    for uid,stream,text,opening in (
        ("123456","private","我下午要准备答辩","上次问过答辩准备得怎么样"),
        ("654321","other","朋友自己的秘密行程","朋友收到的私人开场"),
    ):
        await plugin._store.set_user_stream(uid,stream)
        await plugin._store.record_interaction(uid,text,now-1800,10)
        await plugin._store.add_proactive_pending(uid,uid,"old",stream,now-3600,now-3500)
        await plugin._store.mark_pending_sent(stream,now-3500,event_id=uid)
        await plugin._store.record_proactive_text(uid,opening,now-3500)
    await plugin._store.add_proactive_pending("new","123456","new","private",now,now+120)
    await plugin._store.set_proactive_task_id("new","proactive:life:new")
    active=ActivePluginTask("private","proactive:life:new","proactive","new","new",now,now+600)
    plugin._activate_planner_task=AsyncMock(return_value=active)
    plugin._active_tasks.current=AsyncMock(return_value=active)
    return plugin,active


@pytest.mark.asyncio
@pytest.mark.parametrize("hook",["planner","replyer"])
async def test_only_recipient_context_reaches_both_proactive_models(tmp_path,hook):
    plugin,active=await prepare(tmp_path)
    try:
        before=plugin._store.conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        if hook=="planner":
            result=await plugin.on_planner(session_id="private",items=[
                {"item_type":"SystemMessageItem","parts":[{"type":"text","text":"原有人设"}]},
                {"item_type":"UserMessageItem","parts":[{"type":"text","text":"主动任务"}]},
            ])
        else:
            result=await plugin.on_replyer(session_id="private",extra_prompt="原有人设")
        text=json.dumps(result,ensure_ascii=False)
        assert "我下午要准备答辩" in text and "上次问过答辩准备得怎么样" in text
        assert "朋友自己的秘密行程" not in text and "朋友收到的私人开场" not in text
        assert "原有人设" in text and "+00:00" in text
        assert before==plugin._store.conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        assert (await plugin._store.proactive_event(active.record_id))["status"]=="pending"
    finally:
        await plugin._store.close()


@pytest.mark.asyncio
async def test_other_tasks_sent_turns_and_wrong_stream_cannot_read_recipient_context(tmp_path):
    plugin,active=await prepare(tmp_path)
    try:
        assert not await plugin._proactive_dialogue_note("private",None)
        for kind in ("relay","image"):
            active.kind=kind
            assert not await plugin._proactive_dialogue_note("private",active)
        active.kind="proactive";active.sent=True
        assert not await plugin._proactive_dialogue_note("private",active)
        active.sent=False
        assert not await plugin._proactive_dialogue_note("other",active)
    finally:
        await plugin._store.close()
