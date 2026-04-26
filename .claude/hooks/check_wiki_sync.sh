#!/bin/bash
# Wiki Sync Enforcement — block Stop ถ้า .py ใน ftmo_trading_bot/ ถูกแก้
# แต่ wiki/ + context.md + readme.md + CLAUDE.md ไม่ถูกแตะใน turn เดียวกัน
#
# Mode: decision:block — Claude จะถูกบังคับให้ sync wiki ก่อนจบ turn
# (Stop hook คืน JSON {"decision":"block","reason":...} → ป้องกันการจบ turn → Claude ต้อง sync)

PROJECT_ROOT="/Users/kittipong.n/Desktop/BOT/BOT-FTMO"
cd "$PROJECT_ROOT" 2>/dev/null || exit 0

# ไม่ใช่ git repo → ข้าม
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# รวมทุก change ที่ยังไม่ commit (staged + unstaged + untracked)
CHANGED=$(
  git diff --name-only HEAD 2>/dev/null
  git ls-files --others --exclude-standard 2>/dev/null
)

# มี .py ใน ftmo_trading_bot/ ที่เปลี่ยน?
PY_CHANGED=$(echo "$CHANGED" | grep -E '^ftmo_trading_bot/.+\.py$' | head -10)
if [ -z "$PY_CHANGED" ]; then
  exit 0
fi

# Wiki / context / readme / CLAUDE ถูกแตะหรือไม่?
DOC_CHANGED=$(echo "$CHANGED" | grep -E '^(context\.md|wiki/|readme\.md|CLAUDE\.md)')
if [ -n "$DOC_CHANGED" ]; then
  exit 0
fi

# มี .py เปลี่ยน แต่ doc ไม่เปลี่ยน → block + บังคับให้ sync
PY_LIST=$(echo "$PY_CHANGED" | sed 's/^/  - /')

REASON=$(cat <<EOF
Wiki Sync Protocol violated: Python files were modified but wiki/context/readme were not updated in this turn.

Changed .py files:
${PY_LIST}

Required actions before ending turn (per CLAUDE.md § Wiki Sync Protocol):
1. Identify which wiki sections are affected (obs/config/module/loop/training)
2. Update relevant files: wiki/*.md, context.md, readme.md (if user-facing)
3. Bump 'Last Updated' date on touched docs (YYYY-MM-DD)
4. For obs/architecture/FTMO changes: add entry to wiki/05-invariants.md Version Log

Do NOT skip — incomplete docs cause drift between code and wiki, breaking future-session context.
EOF
)

# Output JSON to stdout — decision:block stops the turn until Claude syncs
# Use python (always present on macOS) for JSON encoding to avoid jq dependency
python3 -c "
import json, sys
reason = sys.stdin.read()
print(json.dumps({
    'decision': 'block',
    'reason': reason,
    'systemMessage': '🔄 Wiki Sync Required — Python edited but wiki/context/readme not updated. Claude must sync before ending turn.'
}))
" <<< "$REASON"

exit 0
