# AwesomeSkills

A curated collection of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) custom slash commands (skills).

精选的 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 自定义斜杠命令（Skills）合集。

[English](#quick-start) | [中文](#快速开始)

---

## Quick Start

```bash
git clone <repo-url> && cd AwesomeSkills
make install
```

All skills will be installed to `~/.claude/commands/` and available as `/skill-name` in any Claude Code session.

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all skills to `~/.claude/commands/` |
| `make uninstall` | Remove all installed skills |
| `make list` | List available skills |

## Skills

<!-- SKILLS_TABLE_START -->
| Skill | Description |
|-------|-------------|
| [`/mentor`](skills/mentor.md) | Project Mentor — generates a complete learning project with skeleton code, comprehensive tests, a study guide, and a structured TODO list. You provide the idea, the mentor builds the scaffolding, you write the core implementation. |
<!-- SKILLS_TABLE_END -->

## Adding a New Skill

1. Create a new `.md` file in `skills/`:

```
skills/my-skill.md
```

2. Write your skill prompt inside the file. The filename becomes the slash command name (e.g., `skills/review.md` -> `/review`).

3. Update this README's Skills table.

4. Run `make install` to apply.

## Skill File Format

Each `.md` file is a prompt template. You can use `$ARGUMENTS` to accept user input:

```markdown
<!-- skills/example.md -->
Analyze the following code and suggest improvements:

$ARGUMENTS
```

Usage in Claude Code: `/example src/main.py`

---

## 快速开始

```bash
git clone <repo-url> && cd AwesomeSkills
make install
```

所有 Skills 会安装到 `~/.claude/commands/`，之后在任意 Claude Code 会话中通过 `/skill-name` 调用。

## 可用命令

| 命令 | 说明 |
|------|------|
| `make install` | 将所有 Skills 安装到 `~/.claude/commands/` |
| `make uninstall` | 移除已安装的 Skills |
| `make list` | 列出所有可用 Skills |

## Skills 列表

<!-- SKILLS_TABLE_CN_START -->
| Skill | 说明 |
|-------|------|
| [`/mentor`](skills/mentor.md) | 项目导师 — 根据你的项目想法，生成完整的学习项目：骨架代码、全面的单元测试、学习指南和结构化的 TODO 列表。导师搭好脚手架，你来写核心实现。 |
<!-- SKILLS_TABLE_CN_END -->

## 添加新 Skill

1. 在 `skills/` 目录下创建 `.md` 文件：

```
skills/my-skill.md
```

2. 在文件中编写 Skill 的 prompt。文件名即为斜杠命令名（如 `skills/review.md` → `/review`）。

3. 更新本 README 的 Skills 表格（中英文各一处）。

4. 执行 `make install` 生效。

## Skill 文件格式

每个 `.md` 文件是一个 prompt 模板，可用 `$ARGUMENTS` 接收用户输入：

```markdown
<!-- skills/example.md -->
分析以下代码并给出改进建议：

$ARGUMENTS
```

在 Claude Code 中使用：`/example src/main.py`

---

## License

MIT
