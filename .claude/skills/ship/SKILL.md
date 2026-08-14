---
name: ship
description: "Ship a change end to end: worktree branch, commits, PR, review, merge to main."
argument-hint: "What are you shipping?"
disable-model-invocation: true
---

# Ship

Five **gates** carry a change from a fresh branch to a merged PR. Each gate ends on a check. A gate opens only when the previous gate's check passes, and the check is a command you run and read — never an assumption that the step went fine.

Resuming mid-flow is the normal case. Locate the gate the repo is actually at before doing anything.

## Locate the gate

```bash
git worktree list && git status --short && git log --oneline origin/main..HEAD
gh pr status
```

State which gate you are at and why, then continue from there:

| What you see | Gate |
| --- | --- |
| On `main`, clean tree | 1 — worktree |
| On a branch, work unfinished or uncommitted | 2 — commit |
| Branch with commits, no PR | 3 — PR |
| PR open | 4 — review |
| PR approved, checks green | 5 — merge |

## Gate 1 — Worktree

Call the `EnterWorktree` tool with `name` set to the branch name. Convention, from `git log`: `<type>/<kebab-summary>`, where type is `feat`, `fix`, `docs`, or `chore` — `fix/arcane-gitops-deploy`, `feat/postgres-only-data`. The tool branches from `origin/main`, so the base is fresh, and it moves the session into the worktree so `main` keeps a clean tree.

**Check:** `git status` reports the new branch with a clean tree, and `git log --oneline -1` matches current `origin/main`.

## Gate 2 — Commit

Commit each coherent unit of work as you finish it, rather than banking one commit at the end. Before each commit, run exactly what CI runs (`.github/workflows/ci-cd.yml`):

```bash
uv run ruff check . && uv run pytest
```

Write the subject imperative and sentence-case, no prefix, one line under ~72 chars — `Close the connection pool when the startup index check fails`. Use a body when the reason for the change is not visible in the diff.

**Check:** lint and tests pass, the tree is clean, and each subject line describes the whole of what its commit changed.

## Gate 3 — PR

```bash
git push -u origin <branch>
gh pr create --base main --title "<subject>" --body "<what and why>"
```

Title reads like the lead commit subject. Body states what changed and why, and names anything the reviewer should look at hardest.

**Check:** `gh pr view` returns an open PR against `main` whose commit list matches `git log origin/main..HEAD`.

## Gate 4 — Review

Two halves, both required:

- **Machine** — `gh pr checks --watch` until every check reports green. A red check sends you back to gate 2.
- **Code** — run `/code-review` over the branch, and read human comments with `gh pr view --comments`.

Every finding ends in one of two states: fixed with a follow-up commit (gate 2 again, including its check), or answered in a PR comment saying why it stands. Report the findings and their resolutions to the user.

**Check:** checks green, and no finding left in neither state.

## Gate 5 — Merge

Merging to `main` is the irreversible step, so ask the user for an explicit go-ahead first, summarising what is about to land.

```bash
gh pr merge <number> --merge
```

`--merge` keeps the `Merge pull request #N from ...` history this repo uses.

Then return the session to `main` and pick up the merge:

```bash
git -C <main-worktree> fetch origin && git -C <main-worktree> pull
```

Call `ExitWorktree` with `action: "remove"` once the branch is merged and nothing uncommitted remains.

**Check:** `gh pr view <number>` reports merged, `main` contains the merge commit, and `git worktree list` no longer shows the worktree.
