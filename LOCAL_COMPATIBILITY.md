# 定制宿主兼容说明

功能差异、对照提交与安装限制见 [LOCAL_CHANGES.md](LOCAL_CHANGES.md)。

- 可通过 `context.relationship_provider="existing_identity"` 接入身份认知与印象卡片，或关闭内部关系温度。
- `linkage.provider="oaimage"` 复用 OAImage 的角色参考图、队列和结果反馈。
- Decision 语义审核、主动历史引用、双向交流与统一关系接口的细节见 `docs/local-*.md`。
- 实际收信人、群聊、主人身份、城市、作息、模型和数据目录均由安装者配置。仓库不携带现有部署的名单与运行数据。
- 运行于 MaiBot 插件监督器，不需要独立服务。离线验证使用 `python -m pytest -q tests`。
