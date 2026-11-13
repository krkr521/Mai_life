"""Semantic review routing, regeneration and actual-send history regressions."""
from datetime import datetime, timedelta
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo
import json
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=util.spec_from_file_location("semantic_test_life",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
package=util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
from semantic_test_life.config import MaiLifeSettings
from semantic_test_life.core.llm_service import LLMService
from semantic_test_life.core.semantic_review import SemanticReview,SimilarityRejected,SimilarityResult
from semantic_test_life.core.storage import LifeStore
from semantic_test_life.life.schedule_service import ScheduleService,hhmm
from semantic_test_life.plugin import MaiLifePlugin

NOW=datetime(2026,11,12,14,0,tzinfo=ZoneInfo("Asia/Shanghai"))
UID="100000001"
OTHER="100000002"
REJECT=SimilarityResult(True,"与昨天的具体转圈趣事过于相近","选择真实的新活动或实质进展，不要只换词")
ACCEPT=SimilarityResult(False,"有不同的具体经历","")


def config():
    return MaiLifeSettings.model_validate({
        "schedule":{"semantic_dedup_enabled":True,"semantic_history_days":3},
        "context":{"outbound_turn_guard":False},
        "users":{"profiles":[{"user_id":UID,"role":"owner"},{"user_id":OTHER}]},
    })


@pytest.mark.asyncio
async def test_strict_llm_preserves_host_timeout_reason_without_duplicate_usage():
    ctx=SimpleNamespace(llm=SimpleNamespace(generate=AsyncMock(return_value={
        "success":False,"error":"decision 模型触发硬超时 10.0s"})),logger=MagicMock())
    store=SimpleNamespace(record_llm_usage=AsyncMock())
    llm=LLMService(ctx,config(),store)
    with pytest.raises(RuntimeError,match="硬超时 10.0s"):
        await llm.generate_json("input","system",None,task_kind="semantic_review",raise_on_failure=True)
    store.record_llm_usage.assert_awaited_once()
    assert "10.0s" in store.record_llm_usage.await_args.kwargs["error_summary"]


@pytest.mark.asyncio
async def test_decision_timeout_retries_identical_candidate_and_history_only_once():
    response={"too_similar":False,"reason":"是新的经历","rewrite_hint":""}
    llm=SimpleNamespace(task_available=lambda _:True,generate_json=AsyncMock(side_effect=[
        RuntimeError("decision 硬超时 10.0s"),response]))
    reviewer=SemanticReview(llm,None,config(),MagicMock())
    result=await reviewer.review("daily_schedule",{"nodes":["新经历"]},[{"summary":"旧趣事"}])
    assert not result.too_similar and llm.generate_json.await_count==2
    assert llm.generate_json.await_args_list[0]==llm.generate_json.await_args_list[1]
    llm.generate_json.side_effect=RuntimeError("decision 硬超时 10.0s")
    llm.generate_json.reset_mock()
    with pytest.raises(RuntimeError,match="10.0s"):
        await reviewer.review("daily_schedule",{"nodes":[]},[{"summary":"旧趣事"}])
    assert llm.generate_json.await_count==2


@pytest.mark.asyncio
async def test_review_routes_to_decision_and_does_not_use_generation_model_override():
    cfg=config();cfg.models.model_override="deepseek-v4-flash-fast"
    async def generate(*,task_name,**kwargs):
        assert task_name=="decision" and kwargs["model"]==""
        return {"success":True,"response":json.dumps({"too_similar":True,"reason":"只是换了说法","rewrite_hint":"换新经历"})}
    ctx=SimpleNamespace(llm=SimpleNamespace(generate=AsyncMock(side_effect=generate)),logger=MagicMock())
    service=LLMService(ctx,cfg);service._supports_task_name=True;service.available_tasks={"decision","utils"}
    reviewer=SemanticReview(service,None,cfg,ctx.logger)
    result=await reviewer.review("proactive_message","姐姐也跟着转了半圈",[{"text":"昨天FUSHI表演转圈"}])
    assert result.too_similar
    assert ctx.llm.generate.await_args.kwargs["task_name"]=="decision"
    assert service.task_for("schedule")=="utils"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad",[None,{},{"too_similar":"false","reason":"text","rewrite_hint":""},
                              {"too_similar":False,"reason":"","rewrite_hint":""}])
async def test_missing_or_malformed_decision_never_approves(bad):
    llm=SimpleNamespace(task_available=lambda _:True,generate_json=AsyncMock(return_value=bad))
    review=SemanticReview(llm,None,config(),MagicMock())
    with pytest.raises(RuntimeError,match="格式无效"):
        await review.review("scene_detail",{"scene":"转圈"},[{"scene":"昨日转圈"}])


@pytest.mark.asyncio
async def test_unavailable_decision_is_reported_instead_of_using_word_matching():
    llm=SimpleNamespace(task_available=lambda _:False,generate_json=AsyncMock())
    review=SemanticReview(llm,None,config(),MagicMock())
    with pytest.raises(RuntimeError,match="不可用"):
        await review.review("daily_schedule",{"nodes":[]},[{"summary":"旧活动"}])
    llm.generate_json.assert_not_awaited()


def raw_nodes(service,day,tag):
    return [{**node,"start":hhmm(node["start_minute"]),"end":hhmm(node["end_minute"]),
             "summary":tag+str(index)} for index,node in enumerate(service._fallback(day,False))]


async def schedule_fixture(tmp_path):
    store=LifeStore(str(tmp_path),journal_mode="delete");await store.initialize()
    llm=SimpleNamespace(task_available=lambda _:True,generate_json=AsyncMock())
    reviewer=SimpleNamespace(require_new=AsyncMock(),update_config=MagicMock())
    service=ScheduleService(store,config(),llm,str(ROOT),MagicMock(),semantic_review=reviewer)
    service.retry_backoff=(0,)
    old=service._fallback((NOW-timedelta(days=1)).date().isoformat(),False)
    old[0]["summary"]="开直播让FUSHI表演转圈"
    await store.replace_framework((NOW-timedelta(days=1)).date().isoformat(),old,"llm")
    return store,llm,reviewer,service


@pytest.mark.asyncio
async def test_generation_sees_exact_review_history_including_late_scenes(tmp_path):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        past=(NOW-timedelta(days=1)).date().isoformat()
        old=service._fallback(past,False)
        for node in old:node["summary"]="上午例行安排"*25
        late=next(node for node in reversed(old) if node["kind"] not in {"sleep","meal","nap"})
        late["summary"]="晚间FUSHI转圈的具体旧趣事"
        await store.replace_framework(past,old,"llm")
        await store.save_scene(late["id"],"尾巴把桌面投影菜单扫乱的旧微场景",{},[])
        llm.generate_json.return_value=raw_nodes(service,str(NOW.date()),"新活动")
        await service.ensure_day(NOW,"虚拟歌姬","晴")
        generation_prompt=llm.generate_json.await_args.args[0]
        history=reviewer.require_new.await_args.args[2]
        assert json.dumps(history,ensure_ascii=False) in generation_prompt
        assert late["summary"] in generation_prompt
        assert "尾巴把桌面投影菜单扫乱的旧微场景" in generation_prompt
        assert "任何具体趣事复用" in generation_prompt
    finally:await store.close()


@pytest.mark.asyncio
async def test_regeneration_accumulates_all_rejection_reasons(tmp_path):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        second_reject=SimilarityResult(True,"散步趣事也重复","改掉旧路线趣事")
        llm.generate_json.side_effect=[raw_nodes(service,str(NOW.date()),tag) for tag in ("旧主题一","旧主题二","新经历")]
        reviewer.require_new.side_effect=[SimilarityRejected(REJECT),SimilarityRejected(second_reject),None]
        await service.ensure_day(NOW,"虚拟歌姬","晴")
        third_prompt=llm.generate_json.await_args_list[2].args[0]
        assert REJECT.reason in third_prompt and second_reject.reason in third_prompt
        assert REJECT.rewrite_hint in third_prompt and second_reject.rewrite_hint in third_prompt
        assert json.dumps(raw_nodes(service,str(NOW.date()),"旧主题二"),ensure_ascii=False) in third_prompt
        assert "只重写重复的具体节点" in third_prompt
    finally:await store.close()


@pytest.mark.asyncio
async def test_daily_regeneration_uses_decision_feedback_before_persisting(tmp_path):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        first=raw_nodes(service,str(NOW.date()),"相似候选")
        second=raw_nodes(service,str(NOW.date()),"不同经历")
        llm.generate_json.side_effect=[first,second]
        reviewer.require_new.side_effect=[SimilarityRejected(REJECT),None]
        nodes=await service.ensure_day(NOW,"虚拟歌姬","晴",force=True)
        assert nodes and all(n["summary"].startswith("不同经历") for n in nodes)
        assert reviewer.require_new.await_count==2
        assert "Decision判定" in llm.generate_json.await_args_list[1].args[0]
        assert REJECT.reason in llm.generate_json.await_args_list[1].args[0]
        assert REJECT.rewrite_hint in llm.generate_json.await_args_list[1].args[0]
    finally:await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure",[RuntimeError("decision offline"),SimilarityRejected(REJECT)])
async def test_failed_semantic_audit_keeps_existing_day_and_never_saves_a_fallback(tmp_path,failure):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        old=service._fallback(str(NOW.date()),False)
        await store.replace_framework(str(NOW.date()),old,"llm")
        previous=await store.get_framework(str(NOW.date()))
        llm.generate_json.return_value=raw_nodes(service,str(NOW.date()),"重复候选")
        reviewer.require_new.side_effect=failure
        with pytest.raises(RuntimeError):
            await service.ensure_day(NOW,"网友","晴",force=True)
        assert await store.get_framework(str(NOW.date()))==previous
    finally:await store.close()


@pytest.mark.asyncio
async def test_scene_rewrites_with_feedback_and_uses_only_past_scene_history(tmp_path):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        llm.generate_json.side_effect=[
            {"scene":"FUSHI又转圈","opportunities":[]},
            {"scene":"在桥边录下自行车铃声作为新采样","opportunities":[]},
        ]
        reviewer.require_new.side_effect=[SimilarityRejected(REJECT),None]
        node={"id":"new-scene","start_minute":840,"end_minute":900,"summary":"录音散步",
              "kind":"leisure","shareability":.8,"energy_load":-1,"location":"桥边"}
        await service.expand_due(NOW,[node],{"energy":70,"hunger":20},"晴")
        saved=await store.get_scene(node["id"])
        assert "自行车铃声" in saved["scene"]
        assert "Decision判定" in llm.generate_json.await_args_list[1].args[0]
        assert all(entry["day"]<str(NOW.date()) for entry in reviewer.require_new.await_args_list[0].args[2])
    finally:await store.close()


@pytest.mark.asyncio
async def test_old_literary_title_is_not_passed_as_an_untyped_music_project(tmp_path):
    store,llm,reviewer,service=await schedule_fixture(tmp_path)
    try:
        old=await store.get_framework(str((NOW-timedelta(days=1)).date()))
        old[0]["summary"]="试唱《播放列表的第七首》的新编曲"
        await store.replace_framework(str((NOW-timedelta(days=1)).date()),old,"llm")
        store.bookshelf_type_hints=AsyncMock(return_value=[
            {"title":"播放列表的第七首","doc_type":"work","work_type":"novel_fragment"},
        ])
        llm.generate_json.return_value=raw_nodes(service,str(NOW.date()),"新活动")
        await service.ensure_day(NOW,"歌姬","晴",force=True)
        prompt=llm.generate_json.await_args.args[0]
        assert "novel_fragment" in prompt and "不得继续继承该错误" in prompt
    finally:await store.close()


async def proactive_fixture(tmp_path):
    p=MaiLifePlugin();p.set_plugin_config(config().model_dump())
    p._ctx=SimpleNamespace(logger=MagicMock())
    p._env=SimpleNamespace(now=lambda:datetime.now(ZoneInfo("Asia/Shanghai")))
    p._store=LifeStore(str(tmp_path),journal_mode="delete");await p._store.initialize()
    await p._store.sync_users(p.config.users.profiles)
    await p._store.set_user_stream(UID,"private")
    await p._store.set_user_stream(OTHER,"other")
    p._semantic_review=SimpleNamespace(review=AsyncMock(return_value=ACCEPT))
    at=time.time()
    await p._store.add_opportunity({"id":"op","framework_id":"","topic":"哼一段秋天的小曲","motive":"分享","weight":1,
                                    "privacy":"normal","expires_at":at+7200})
    for event,user,stream,stamp in (("old",UID,"private",at-86400),("other-old",OTHER,"other",at-86400)):
        await p._store.add_proactive_pending(event,user,"op",stream,stamp,stamp+120)
        assert await p._store.mark_pending_sent(stream,stamp,event_id=event)
    await p._store.record_proactive_text("old","昨天刚让FUSHI表演转圈",at-86400)
    await p._store.record_proactive_text("other-old","其他用户的私人话题",at-86400)
    await p._store.add_proactive_pending("new",UID,"op","private",at,at+120)
    await p._store.set_proactive_task_id("new","proactive:new")
    return p


@pytest.mark.asyncio
async def test_actual_reply_not_opportunity_title_is_reviewed_and_only_its_user_history_is_used(tmp_path):
    p=await proactive_fixture(tmp_path)
    try:
        p._semantic_review.review.return_value=REJECT
        result=await p.on_semantic_replyer_after(session_id="private",reply_message_id="proactive:new",
                                                response="FUSHI转圈，姐姐也跟着转了半圈",retry_count=0,max_retries=3)
        value=result["modified_kwargs"]
        assert value["retry"] and value["response"]==""
        assert REJECT.reason in value["retry_reason"]
        candidate=p._semantic_review.review.await_args.args[1]
        assert "姐姐也跟着" in candidate["response"]
        history=p._semantic_review.review.await_args.args[2]
        assert len(history)==1 and "其他用户" not in json.dumps(history,ensure_ascii=False)
        await p.on_replyer_after(**value)
        assert not p._reply_confirmations
        assert (await p._store.proactive_event("new"))["status"]=="pending"
    finally:await p._store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode",["exhausted","offline"])
async def test_exhausted_or_failed_audit_cancels_without_deducting_quota(tmp_path,mode):
    p=await proactive_fixture(tmp_path)
    try:
        if mode=="offline":p._semantic_review.review.side_effect=RuntimeError("offline")
        else:p._semantic_review.review.return_value=REJECT
        previous=(await p._store.get_user(UID))["proactive_count"]
        result=await p.on_semantic_replyer_after(session_id="private",reply_message_id="proactive:new",
                                                response="FUSHI又转圈",retry_count=3,max_retries=3)
        assert not result["modified_kwargs"]["response"]
        assert (await p._store.proactive_event("new"))["status"]=="cancelled"
        assert (await p._store.get_user(UID))["proactive_count"]==previous
        assert not p._reply_confirmations
    finally:await p._store.close()


@pytest.mark.asyncio
async def test_normal_reply_and_other_plugin_completion_do_not_enter_the_audit(tmp_path):
    p=await proactive_fixture(tmp_path)
    try:
        for anchor in ("123456","proactive:a-dawn.oaimage:job"):
            result=await p.on_semantic_replyer_after(session_id="private",reply_message_id=anchor,
                                                    response="同样提到FUSHI也可以回答用户的问题",max_retries=3)
            assert result=={"action":"continue"}
        p._semantic_review.review.assert_not_awaited()
    finally:await p._store.close()


@pytest.mark.asyncio
async def test_only_successful_real_send_chunks_are_persisted_and_duplicates_are_idempotent(tmp_path):
    p=await proactive_fixture(tmp_path)
    try:
        args={"session_id":"private","reply_message_id":"proactive:new","response":"在桥边录了新的铃声"}
        assert await p.on_semantic_replyer_after(**args)=={"action":"continue"}
        await p.on_replyer_after(**args)
        def message(text):
            return {"session_id":"private","raw_message":[{"type":"text","data":text}],
                    "message_info":{"additional_config":{"mai_life_active_task_id":"proactive:new"}}}
        await p.on_send_after(message=message("没发出去"),reply_message_id="proactive:new",sent=False)
        assert len(await p._store.recent_proactive_text(UID,time.time()-3*86400))==1
        # Model-generated but failed output is not in the ledger. Simulate a later successful retry.
        await p.on_send_after(message=message("在桥边录了新的铃声"),reply_message_id="proactive:new",sent=True)
        await p.on_send_after(message=message("声音像一颗小星星"),reply_message_id="proactive:new",sent=True)
        await p.on_send_after(message=message("声音像一颗小星星"),reply_message_id="proactive:new",sent=True)
        rows=await p._store.recent_proactive_text(UID,time.time()-3*86400)
        assert len(rows)==2 and "没发出去" not in rows[-1]["text"]
        assert rows[-1]["text"].count("小星星")==1
    finally:await p._store.close()


@pytest.mark.asyncio
async def test_unconfirmed_event_text_is_not_treated_as_delivered(tmp_path):
    p=await proactive_fixture(tmp_path)
    try:
        assert not await p._store.record_proactive_text("new","只是生成草稿",time.time())
        assert not await p._store.record_proactive_text("missing","不存在的任务",time.time())
        rows=await p._store.recent_proactive_text(UID,time.time()-3*86400)
        assert len(rows)==1
    finally:await p._store.close()
