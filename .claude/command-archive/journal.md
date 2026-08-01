---
description: ⚡ Write some journal entries.
---

## Writing Style Detection

Before writing, check for a writing-styles directory:
1. Check `docs/writing-styles/` — if exists, read all `.md` files inside
2. Check `assets/writing-styles/` — if exists and step 1 not found, read all `.md` files inside
3. If writing styles found → adopt tone, vocabulary, sentence structure, and formatting rules from those files
4. If no writing-styles directory found → write freely in a natural, conversational tone

## Journal Writing

Use the `journal-writer` subagent to explore the memories and recent code changes, and write journal entries.
- Concise, focused on important events, key changes, impacts, and decisions
- Apply detected writing style (if any) to the journal voice
- Keep journal entries in `docs/journal/` directory