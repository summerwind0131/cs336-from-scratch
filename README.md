# CS336 from scratch

CS336 Spring 2025 的作业与学习记录，统一保存在一个 Git 仓库中。

| 目录 | 内容 |
| --- | --- |
| [assignment1-basics](assignment1-basics/) | 作业 1：Basics，保留原有实现和提交历史 |
| [assignment2-systems](assignment2-systems/) | 作业 2：Systems，基于官方 `spring2025` 标签 |

各作业独立管理 Python 环境。请进入对应目录后使用 `uv run`，具体设置见各自的 README 和作业讲义。

## 官方来源

- 作业 1：<https://github.com/stanford-cs336/assignment1-basics>
- 作业 2：<https://github.com/stanford-cs336/assignment2-systems/tree/spring2025>

作业 2 通过 `git subtree` 导入，目录中的文件由本仓库直接跟踪，无需初始化 submodule。
其 `cs336-basics` 目录保留官方随附的实现；当前没有改为依赖自己的作业 1。

## Git 操作

在本目录统一提交和推送两份作业的改动：

```sh
git status
git add <修改的文件>
git commit -m "描述修改"
git push origin main
```

本机的 `origin` 指向个人仓库 `summerwind0131/cs336-from-scratch`，
`upstream` 指向官方作业 1 仓库，`upstream-assignment2` 指向官方作业 2 仓库。
这两个官方仓库各自只包含一份作业；更新时应按对应目录导入。
