#!/bin/bash
# Appends a timestamped entry to PROGRESS.md when a Claude session ends.
# Triggered by the Stop hook in .claude/settings.json.

PROGRESS_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/PROGRESS.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "no commits yet")

if [ ! -f "$PROGRESS_FILE" ]; then
  exit 0
fi

# Avoid duplicate entries — skip if last line already has today's date
TODAY=$(date "+%Y-%m-%d")
if tail -5 "$PROGRESS_FILE" | grep -q "$TODAY"; then
  exit 0
fi

cat >> "$PROGRESS_FILE" << EOF

---
**$TIMESTAMP**
Last commit: $LAST_COMMIT
EOF

exit 0
