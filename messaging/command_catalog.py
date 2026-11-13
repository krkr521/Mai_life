"""Mai_life 指令菜单的结构化目录与纯文本降级内容。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class CommandItem:
    command:str
    description:str


@dataclass(frozen=True,slots=True)
class CommandSection:
    title:str
    items:tuple[CommandItem,...]


COMMAND_SECTIONS:tuple[CommandSection,...]=(
    CommandSection("日常状态",(
        CommandItem("/八千代状态","生活状态、模型与后台任务健康"),
        CommandItem("/八千代配置","当前插件配置摘要"),
        CommandItem("/八千代主动私聊 [开启|关闭|状态]","查询或修改自己的主动私聊开关（需用户档案）"),
        CommandItem("/八千代群聊摘要 [开启|关闭|状态]","查询或修改自己的群聊摘要订阅（需用户档案）"),
        CommandItem("/八千代主动额度 [0-20]","查看或修改自己的每日主动私聊额度（需用户档案）"),
        CommandItem("/八千代免打扰 [HH:MM-HH:MM|关闭]","查看、设置或关闭自己的主动私聊免打扰（需用户档案）"),
    )),
    CommandSection("生活记录",(
        CommandItem("/八千代重要日期 [页码]","查看自己的重要日期和待确认候选（需用户档案）"),
        CommandItem("/八千代添加日期 日期 名称","添加自己的重要日期（需用户档案）"),
        CommandItem("/八千代删除日期 名称 [日期]","按自己取的名字删除重要日期，同名可补日期（需用户档案）"),
        CommandItem("/八千代确认日期 编号 [日期]","确认日期候选（需用户档案）"),
    )),
    CommandSection("见闻与书柜",(
        CommandItem("/八千代转述 群QQ号 内容","向白名单群发起简单转述"),
    )),
    CommandSection("管理与诊断",(
        CommandItem("/八千代立即创作","立即执行创作判断（管理员）"),
        CommandItem("/八千代重生日程","重新生成今日日程（管理员）"),
        CommandItem("/八千代休息测试","休息闸门诊断（管理员）"),
    )),
)


def build_command_usage_text(notice:str="")->str:
    """构造不依赖 Markdown 的命令菜单文本，命令菜单的唯一发送形态。"""
    lines=["麦生活 · 指令中心","输入 /八千代 或 /八千代帮助 可再次查看菜单。"]
    clean_notice=" ".join(str(notice or "").replace("\x00","").split())[:120]
    if clean_notice:lines.extend(("",clean_notice))
    for section in COMMAND_SECTIONS:
        lines.extend(("",section.title))
        lines.extend(f"{item.command}  {item.description}" for item in section.items)
    lines.extend(("","私聊管理员无需重复创建用户档案；重要日期操作仍需启用用户档案。"))
    return "\n".join(lines)


__all__=["COMMAND_SECTIONS","CommandItem","CommandSection","build_command_usage_text"]
