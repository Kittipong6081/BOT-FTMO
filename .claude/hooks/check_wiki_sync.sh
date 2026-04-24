#!/bin/bash
# Wiki Sync Check — เตือนถ้ามี .py ที่ถูกแก้ใน ftmo_trading_bot/ แต่ wiki/context/readme ไม่ถูกแตะ
# ทำงานเป็น Stop hook (เรียกตอนจบ turn) — ไม่บล็อก, แค่ warn ทาง stderr

PROJECT_ROOT="/Users/kittipong.n/Desktop/BOT/BOT-FTMO"
cd "$PROJECT_ROOT" 2>/dev/null || exit 0

# ไม่ใช่ git repo → ข้าม
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# รวมทุก change (staged + unstaged + untracked)
CHANGED=$(
  git diff --name-only HEAD 2>/dev/null
  git ls-files --others --exclude-standard 2>/dev/null
)

# มี .py ใน ftmo_trading_bot/ ที่เปลี่ยน?
PY_CHANGED=$(echo "$CHANGED" | grep -E '^ftmo_trading_bot/.+\.py$' | head -5)
if [ -z "$PY_CHANGED" ]; then
  exit 0
fi

# Wiki / context / readme / CLAUDE ถูกแตะหรือไม่?
DOC_CHANGED=$(echo "$CHANGED" | grep -E '^(context\.md|wiki/|readme\.md|CLAUDE\.md)')
if [ -n "$DOC_CHANGED" ]; then
  exit 0
fi

# มี .py เปลี่ยน แต่ doc ไม่เปลี่ยน → warn
echo "⚠️  Wiki Sync Protocol: Python files changed but wiki/context/readme NOT updated" >&2
echo "   Changed .py files:" >&2
echo "$PY_CHANGED" | sed 's/^/     - /' >&2
echo "   → อัพเดท wiki/ + context.md + readme.md ตาม CLAUDE.md ก่อน commit" >&2
exit 0
