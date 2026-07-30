---
name: git-workflow
description: Create, commit, and push small atomic changes in this repository. Use whenever Codex implements a feature, fixes a bug, refactors code, changes tests, modifies build tooling, or changes documentation; split a larger feature into independently reviewable and reversible commits, stage only the current unit, use English Conventional Commit messages, and push each commit to the configured GitHub remote.
---

# Git Workflow

## Project profile: Embedding Runtime Android

- Application modules: `:app` (sentence embeddings demo) and `:app-model2vec` (model2vec demo).
- Native library modules: `:sentence_embeddings` wraps `rs-hf-tokenizer`; `:model2vec` wraps `rs-model2vec`. Both use Rust and require Android NDK `28.1.13356709` (r28b).
- `:app` runs `downloadModelsIfNeeded` from `preBuild`; an application assembly can download large ONNX model assets when they are absent. Do not make that download an implicit side effect of routine validation.
- Treat model assets, Rust build output, Android build output, local SDK paths, keystores, and credentials as non-source artifacts. Never stage them unless the user explicitly requests a reviewed release asset change.

Apply this workflow after each completed atomic change. A feature is often a sequence of commits, not automatically one commit. A commit may be a behavior slice, a focused refactor, a test addition, a build/tooling update, or a documentation update. Do not defer all Git work until the end of a multi-step task.

## 1. Map atomic commit units before staging

For work that contains more than one intent, write a short commit map before editing or staging. Describe each unit by the question a reviewer should be able to answer from its diff, for example: "Does the local-model store delete a downloaded model safely?" or "Does the settings screen expose the completed delete operation?"

Use a separate commit when all of these are true:

- It has one clear review intent and a concise commit message.
- It can be reverted without corrupting data, leaving an exposed non-working action, or requiring unrelated changes to be reverted.
- Its validation can be stated independently.
- It is not merely a partial file-level fragment of another unit.

Keep changes together when splitting would leave either commit invalid, misleading, or unusable. In particular, do not separate an API change from its required call-site migration, a persistence schema change from its mandatory migration, or a UI action from the only code that makes that action work. Do not split simply because files live in different layers.

Use these default boundaries:

| Change | Default boundary |
|---|---|
| New behavior | One commit per independently usable behavior slice; include its required wiring and resources. |
| Refactor that enables later behavior | Separate `refactor` commit only when it has no behavior change and remains valid on its own. |
| Tests | Keep tests with the behavior they prove. Use a separate `test` commit only for independent coverage of already-present behavior. |
| README, guides, changelogs | Use a separate `docs` commit after the documented behavior is implemented and verified, unless the document itself is the requested deliverable. |
| Build, generated configuration, dependency lockfiles | Keep with the code that requires them; use `chore` only when they are independently useful. |

Example commit map for "allow users to delete a downloaded local model":

1. `refactor: expose local model storage operations` — only if extracting the storage boundary changes no behavior and is independently verified.
2. `feat: delete downloaded local model files` — deletion rules, state update, and focused tests needed to make deletion safe.
3. `feat: add local model delete action` — the completed UI/view-model wiring, confirmation and error handling needed for a working user action.
4. `docs: document local model deletion` — README or user-facing guide updates after the feature works.

Omit any unit that is not needed. Merge units 1–3 when their code is inseparable or intermediate commits would violate project rules. Never create cosmetic or knowingly broken commits merely to increase the count.

## 2. Define and inspect the current boundary

Use one commit for one independently reviewable and reversible behavior change. Include directly related production code, tests, resources, and documentation in that commit. Split unrelated changes into separate commits.

Before staging, run:

```powershell
git status --short
git diff -- <paths-touched>
```

Preserve pre-existing user changes. Identify the files created or modified for the current change and stage only those paths. Never use `git add .`, `git add -A`, or `git commit -a` in a dirty worktree.

If a current change overlaps an unowned user modification, stop and ask for direction rather than staging it accidentally.

## 3. Verify before committing

Run the smallest relevant verification before each code commit. Choose from this project-specific matrix:

| Changed scope | Default verification |
|---|---|
| Documentation, workflow Skill, or ignore-rule only | `git diff --check`; inspect the changed file(s). |
| Kotlin/Compose in `:app` | `.\gradlew.bat :app:compileDebugKotlin` and focused unit tests when present. |
| Kotlin/Compose in `:app-model2vec` | `.\gradlew.bat :app-model2vec:compileDebugKotlin` and focused unit tests when present. |
| `:sentence_embeddings`, `rs-hf-tokenizer`, or their build configuration | `.\gradlew.bat :sentence_embeddings:assembleDebug` |
| `:model2vec`, `rs-model2vec`, or their build configuration | `.\gradlew.bat :model2vec:assembleDebug` |
| Shared build configuration | Run the relevant native-library assemblies plus the affected app compile task. |

Run `:app:assembleDebug` only when the application artifact itself is the validation target or model assets are known to be present; otherwise it may download models. Run a fuller lint or test task when the changed behavior warrants it. Inspect failures before committing.

Do not commit or push a known regression. If verification is blocked by a clearly pre-existing or environmental failure, report the command and evidence accurately, attempt a safe targeted alternative, and ask before committing code that cannot be verified.

## 4. Stage and inspect the exact change

Stage only the files belonging to the completed change:

```powershell
git add -- <path-1> <path-2>
git diff --cached --check
git diff --cached
```

Confirm the staged diff contains no credentials, generated build output, model binaries, user media, or unrelated edits. Unstage an accidentally staged path with:

```powershell
git restore --staged -- <path>
```

Do not alter, discard, or stage any other worktree changes.

## 5. Commit with an English Conventional Commit message

Use exactly one of these lower-case types followed by a colon, a space, and a concise English imperative summary:

| Type | Use for | Example |
|---|---|---|
| `feat` | New user-visible capability | `feat: add image OCR ingestion` |
| `fix` | Bug correction | `fix: reopen cached URL input stream` |
| `docs` | Documentation-only change | `docs: add multimodal development guide` |
| `style` | Formatting-only change with no runtime effect | `style: format document import code` |
| `refactor` | Code restructuring without feature or bug behavior | `refactor: isolate content ingestion orchestration` |
| `test` | Tests only | `test: cover Chinese text chunking` |
| `chore` | Build, dependency, tooling, or auxiliary maintenance | `chore: configure objectbox test task` |

Commit with the selected message:

```powershell
git commit -m "<type>: <concise English summary>"
```

Do not use generic messages such as `update`, `fix bug`, `changes`, or `wip`. Do not amend, squash, rebase, reset, or rewrite history unless the user explicitly asks.

## 6. Push immediately after a successful commit

Confirm the current branch and push remote before pushing:

```powershell
git branch --show-current
git remote get-url --push origin
git push origin HEAD
```

Push every successful commit immediately. If the push is rejected, fails authentication, or fails network delivery, do not force-push, retry with destructive history changes, or claim the commit reached GitHub. Report the local commit SHA, branch, exact failure, and the remaining safe next step.

## 7. Confirm the boundary and continue

After a successful push, run:

```powershell
git status --short
git log -1 --oneline
```

Report the commit SHA, message, branch, push result, verification evidence, and any intentionally preserved user changes. If the commit map has another completed unit, repeat this workflow for that unit; otherwise confirm the final worktree state. Do not create branches, pull requests, tags, or releases unless the user separately requests them.
