"""A reloaded life plugin must recover a real, scoped reply reference without inbound authority."""
from unittest.mock import AsyncMock
from maibot_sdk.context import PluginContext

import asyncio
import base64
import hashlib
import time

import pytest

from test_recipient_context import prepare
from mailife_local.messaging.task_context import planner_has_message_reference,planner_includes_private_message


def history(message_id):
    display=message_id
    if len(display)>12:
        display="m"+base64.b32encode(hashlib.sha1(display.encode()).digest()).decode().lower()[:6]
    return [
        {"item_type":"UserMessageItem","parts":[{"type":"text","text":
            f'<message msg_id="{display}" user="测试用户">早上的对话</message>\n'
            f'<message msg_id="{display}" user="测试用户" boundary="end" />'}]},
        {"item_type":"UserMessageItem","parts":[{"type":"text","text":
            '<plugin_proactive_task id="proactive:life:new" plugin_id="maibot-community.mai-life">本轮主动任务</plugin_proactive_task>'}]},
    ]


def source(message_id):
    return {"message_id":message_id,"session_id":"private","is_notify":False,
            "message_info":{"user_info":{"user_id":"123456"},"group_info":None}}


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id",["-898104413","snowluma-long-verified-message-1000000012"])
async def test_reload_restores_raw_id_from_durable_record_and_current_host_history(tmp_path,message_id):
    plugin,active=await prepare(tmp_path)
    try:
        await plugin._store.record_interaction("123456","早上的对话",time.time(),10,source_message_id=message_id)
        await plugin._store.record_interaction("654321","另一个人的对话",time.time()+1,10,source_message_id="other-id")
        rpc=AsyncMock(return_value={"success":True,"message":source(message_id)})
        plugin._ctx=PluginContext("maibot-community.mai-life",rpc_call=rpc)
        items=history(message_id)
        assert planner_has_message_reference(items,message_id)
        assert not planner_includes_private_message(items,message_id)
        before=dict(plugin._session_runtime["private"])
        assert await plugin._planner_reply_anchor("private",active,items)==message_id
        rpc.assert_awaited_once()
        assert rpc.await_args.args[2]["capability"]=="message.get_by_id"
        assert rpc.await_args.args[2]["args"]["message_id"]==message_id
        assert rpc.await_args.args[2]["args"]["chat_id"]=="private"
        assert plugin._session_runtime["private"]==before
        assert "quiet_turn_token" not in plugin._session_runtime["private"]
        result=await plugin.on_planner(session_id="private",items=items)
        text=str(result)
        assert f"msg_id 设为 {message_id}" in text and "本轮没有可引用" not in text
        assert "不重答旧消息" in text
    finally:
        await plugin._store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad",["sender","session","group","message_id","notify","missing"])
async def test_recovery_rejects_unverified_foreign_or_notification_messages(tmp_path,bad):
    plugin,active=await prepare(tmp_path)
    try:
        mid="-898104413"
        await plugin._store.record_interaction("123456","早上的对话",time.time(),10,source_message_id=mid)
        message=source(mid)
        if bad=="sender":message["message_info"]["user_info"]["user_id"]="654321"
        elif bad=="session":message["session_id"]="other"
        elif bad=="group":message["message_info"]["group_info"]={"group_id":"11111"}
        elif bad=="message_id":message["message_id"]="other-id"
        elif bad=="notify":message["is_notify"]=True
        result={"success":True,"message":None if bad=="missing" else message}
        plugin._ctx=PluginContext("maibot-community.mai-life",rpc_call=AsyncMock(return_value=result))
        assert not await plugin._planner_reply_anchor("private",active,history(mid))
        assert "message_id" not in plugin._session_runtime["private"]
    finally:
        await plugin._store.close()


@pytest.mark.asyncio
async def test_absent_host_reference_is_not_invented_or_reintroduced_after_context_clear(tmp_path):
    plugin,active=await prepare(tmp_path)
    try:
        await plugin._store.record_interaction("123456","已清理的旧对话",time.time(),10,source_message_id="old-id")
        rpc=AsyncMock()
        plugin._ctx=PluginContext("maibot-community.mai-life",rpc_call=rpc)
        assert not await plugin._planner_reply_anchor("private",active,history("different-id"))
        rpc.assert_not_awaited()
    finally:
        await plugin._store.close()


@pytest.mark.asyncio
async def test_resumed_service_checks_once_before_the_regular_patrol_sleep(tmp_path):
    plugin,active=await prepare(tmp_path)
    entered=asyncio.Event()
    async def patrol(now,state):
        entered.set()
        return False
    plugin._proactive=type("Proactive",(),{})()
    plugin._proactive.patrol=AsyncMock(side_effect=patrol)
    plugin._resolve_all_streams=AsyncMock()
    task=asyncio.create_task(plugin._proactive_loop())
    try:
        await asyncio.wait_for(entered.wait(),timeout=1)
        assert plugin._proactive.patrol.await_count==1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        await plugin._store.close()
