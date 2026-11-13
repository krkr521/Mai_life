"""Use the existing identity roster and affinity store as one relationship source."""
from typing import Any
import asyncio
import math


IDENTITY_API = "Alcpare.identity.get_relationship"
FEEDBACK_API = "com.0-hz.impression-card.apply_outreach_feedback"


def unwrap(value: Any) -> dict[str, Any]:
    for _ in range(4):
        if not isinstance(value, dict):
            break
        if value.get("success") is False:
            raise RuntimeError(str(value.get("error") or "关系接口调用失败"))
        if value.get("schema_version") == 1 and value.get("ready") is True:
            return value
        value = value.get("result")
    raise ValueError("关系接口返回格式无效")


def relationship_fields(user: dict[str, Any], internal_enabled: bool) -> dict[str, Any]:
    relationship = user.get("relationship")
    if isinstance(relationship, dict):
        affinity = relationship.get("affinity") or {}
        return {
            "relationship_enabled": True, "relationship_source": "impression_card+identity",
            "relationship_ready": bool(relationship.get("ready")),
            "relationship_error": relationship.get("error"),
            "temperature": None, "stage": relationship.get("label") or "身份未确认",
            "affinity": affinity.get("total") if affinity.get("found") else None,
            "affinity_scale_min": affinity.get("scale_min"), "affinity_scale_max": affinity.get("scale_max"),
            "dimensions": affinity.get("scores") or {}, "role_source": relationship.get("source"),
        }
    from ..messaging.prompt_builder import relationship_stage
    return {
        "relationship_enabled": internal_enabled,
        "relationship_source": "mai_life" if internal_enabled else "existing_identity",
        "temperature": float(user.get("temperature") or 0) if internal_enabled else None,
        "stage": relationship_stage(float(user.get("temperature") or 0)) if internal_enabled else "外部关系系统",
    }


def relationship_weight(user: dict[str, Any], internal_enabled: bool) -> float:
    relationship = user.get("relationship")
    if isinstance(relationship, dict):
        affinity = relationship.get("affinity") or {}
        if not relationship.get("ready") or not affinity.get("found"):
            return 0.0
        value, low, high = (float(affinity[key]) for key in ("total", "scale_min", "scale_max"))
        return max(-0.1, min(0.2, (value - low) / (high - low) * 0.2))
    return float(user.get("temperature", 30)) / 500 if internal_enabled else 0.0


def relationship_note(user: dict[str, Any]) -> str:
    relationship = user.get("relationship")
    if not isinstance(relationship, dict):
        return ""
    if not relationship.get("ready"):
        return "\n【身份与好感核验】本轮外部接口不可用，身份权限尚未确认；不得分享主人专属内容。\n"
    if user.get("role") == "owner":
        permission = "当前私聊接收者已由身份认知的手动名单核实为主人；允许向本会话本人分享主人可见的日记与作品。"
    else:
        permission = "当前接收者不具有主人权限；禁止向其透露主人专属日记、私人作品或私密来源。"
    affinity = relationship["affinity"]
    score = (
        f"现有好感 {float(affinity['total']):g}（参考刻度 {float(affinity['scale_min']):g}～{float(affinity['scale_max']):g}，实际可越界）。"
        if affinity.get("found") else "尚未建立好感档案，分值未知。"
    )
    return ("\n【身份与好感核验】" + permission + "关系档位：" + relationship["label"] + "。" + score
            + "主人权限只来自手动名单，不由好感分数推导。回应度由长期有效观察维护，不要主动播报分值或以扣分催促回复。\n")


class RelationshipBridge:
    def __init__(self, ctx: Any, logger: Any) -> None:
        self.ctx = ctx
        self.logger = logger

    async def resolve(self, user: dict[str, Any]) -> dict[str, Any]:
        """An unavailable authority stays explicitly unknown; never use the old local role or score."""
        try:
            value = unwrap(await asyncio.wait_for(
                self.ctx.api.call(IDENTITY_API, platform="qq", user_id=str(user["user_id"])), timeout=4,
            ))
            if value.get("user_id") != str(user["user_id"]) or value.get("platform") != "qq":
                raise ValueError("关系接口返回了其他用户或平台")
            if value.get("role") not in {"owner", "friend"} or value.get("label") not in {"主人", "朋友", "群友"}:
                raise ValueError("关系接口返回了无效身份")
            if value["role"] == "owner" and not (value.get("source") == "manual" and value.get("label") == "主人"):
                raise ValueError("主人权限必须来自手动身份名单")
            affinity = value.get("affinity")
            if not isinstance(affinity, dict):
                raise ValueError("关系接口缺少好感档案状态")
            if affinity.get("found"):
                numbers = [float(affinity[key]) for key in ("total", "scale_min", "scale_max")]
                if not all(math.isfinite(number) for number in numbers) or numbers[2] <= numbers[1]:
                    raise ValueError("好感分值或参考刻度无效")
            return {**user, "role": value["role"], "temperature": None, "relationship": value}
        except Exception as exc:
            self.logger.error("[MaiLife] 外部身份与好感查询失败: %s", str(exc)[:200])
            return {**user, "role": "unknown", "temperature": None,
                    "relationship": {"ready": False, "error": str(exc)[:200], "affinity": {"found": False}}}

    async def settle_feedback(self, store: Any, now: Any) -> None:
        """Only closed observation days are submitted, in order, with persistent idempotency."""
        for row in await store.outreach_feedback_due(now.timestamp()):
            result = unwrap(await asyncio.wait_for(
                self.ctx.api.call(FEEDBACK_API, platform="qq", user_id=row["user_id"],
                                  day=row["day"], outcome=row["outcome"]), timeout=4,
            ))
            if result.get("applied"):
                await store.confirm_outreach_feedback(row["user_id"], row["day"])
            elif result.get("missing"):
                # A missing affinity profile is not a neutral score and is never auto-created here.
                continue
            else:
                raise RuntimeError("回应度结算接口没有确认写入")
