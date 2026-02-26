# Active Context Protocol

External context management for Claude Code agents.

## What is ACP?

Claude Code agents have no awareness of their own context window size. They work until the window fills, and by that point it's too late for clean compaction -- the agent is already losing coherence or hitting limits mid-task.

ACP is an external monitor that solves this. It watches the agent's JSONL session log from outside, reads token usage from each API call entry, and delivers reminder messages via tmux when thresholds are crossed. It also detects work milestones (PR merges, issue closures, commits) and reminds the agent to file learnings to institutional memory.

The key design principle: **ACP sends a reminder, not an auto-trigger.** The agent decides when and how to act. ACP just makes sure the agent has the information it needs to make that decision.

## Features

- **Token monitoring** with configurable threshold (default 70,000 tokens)
- **Compaction reminders** with grace period, cooldown, and stacking prevention
- **Memory filing reminders** -- detects PR merges, issue closures, commits, and PR creates via pattern matching on tool results and tool calls
- **Two delivery modes**: `reminder` (default, delivers anytime) and `command` (requires idle detection -- for future `/compact`-style delivery)
- **tmux delivery** using the proven two-call `send-keys` pattern (text first, Enter separately)
- **Warmdown enforcement** -- global rate limit prevents reminder spam
- **Session tracking** -- auto-detects the active JSONL session file, handles rotation
- **Incremental reads** -- seeks from last position, never re-reads the full log
- **YAML configuration** with zero-dependency fallback parser (PyYAML optional)
- **Full CLI**: `start`, `stop`, `status`, `config`, `init`
- **Audit trail** logging for all delivery attempts (delivered, queued, skipped, failed)
- **PID file management** for clean start/stop lifecycle

## Requirements

- Python 3.11+
- tmux (for delivery)
- Claude Code (the agent being monitored)
- No external Python dependencies (PyYAML is optional for config parsing)

## Quick Start

```bash
# Install
pip install -e .
# or: pipx install -e .

# Create default config
acp init
# -> writes ~/.acp/config.yaml

# Edit config: set tmux_session to your Claude Code session name
# e.g., tmux_session: "claude-0"

# Start the monitor
acp start

# Check if it's running
acp status

# View resolved config
acp config

# Stop when done
acp stop
```

## Configuration

ACP reads from `~/.acp/config.yaml`. Run `acp init` to generate the default:

```yaml
# Token monitoring -- how often to check and when to alert
token_threshold: 70000        # Total context tokens before compaction reminder
polling_interval: 30          # Seconds between monitoring cycles

# Delivery -- how reminders reach the agent
warmdown_interval: 120        # Minimum seconds between any two reminders (global)
grace_period: 300             # Seconds after session start before any reminders fire
tmux_session: ""              # Target tmux session name (REQUIRED for delivery)
log_file: "~/.acp/acp.log"   # Audit trail for all delivery attempts

# Idle detection
idle_threshold: 5.0           # Seconds since last JSONL write to consider agent idle

# Compaction trigger
compaction:
  enabled: true               # Whether compaction reminders are active
  threshold: 70000            # Token count that triggers compaction reminder
  cooldown: 120               # Seconds between compaction reminders

# Memory filing trigger
memory_filing:
  enabled: true               # Whether memory filing reminders are active
  grace_after_event: 60       # Seconds to wait after milestone before reminding
  patterns: []                # Additional regex patterns for milestone detection
```

The only required change is `tmux_session` -- set it to the name of the tmux session where Claude Code is running. Everything else has sensible defaults.

If PyYAML is not installed, ACP uses a built-in minimal YAML parser that handles the flat key-value format above. No external dependencies required.

## Architecture

```
Claude Code Agent (tmux session)
    |  writes
    v
JSONL Session Log (~/.claude/projects/<project-path>/<session-id>.jsonl)
    ^  reads (incremental, seek-from-last-position)
    |
ACP Monitor (polling loop, configurable interval)
    |
    +-- FileTracker ---------> finds active session JSONL, detects rotation
    +-- TokenMonitor --------> reads token usage from assistant entries
    +-- CompactionTrigger ---> threshold / grace / cooldown / stacking checks
    +-- MemoryFilingTrigger -> pattern-matches milestones in tool results & calls
    +-- DeliverySystem ------> sends reminder via tmux send-keys (two-call pattern)
```

**How it reads tokens**: Each `assistant` entry in the JSONL contains a `message.usage` block with `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`. ACP sums these for total context. This value grows monotonically, so only the latest entry matters.

**How it delivers**: ACP sends text to the tmux session via `tmux send-keys`. Two separate subprocess calls are required -- one for the message text, one for Enter -- because a single call silently drops the Enter keystroke. In `reminder` mode (default), delivery happens regardless of agent activity. In `command` mode, ACP waits for the agent to be idle (no recent JSONL writes, last entry is an `assistant` type without `tool_use` blocks).

**How it prevents spam**: Four layers of protection:
1. **Grace period** -- no reminders during the first N seconds of a session
2. **Cooldown** -- minimum time between consecutive reminders from the same trigger
3. **Stacking prevention** -- won't re-fire if a reminder is pending and cooldown hasn't expired
4. **Warmdown** -- global rate limit across all triggers

## What the Agent Sees

When ACP delivers a compaction reminder:

```
[ACP] Context at 85000 tokens (121% of 70000 threshold). Consider compacting when ready.
```

When ACP delivers a memory filing reminder:

```
[ACP] Significant work detected. Consider filing to memory: takeaways, decisions, facts, or proverbs.
```

These arrive as text input in the tmux session. The agent processes them on its next turn.

## Use Cases

### Shipped

**Compaction reminders** -- Get nudged when context grows large. The agent can compact immediately, finish its current task first, or ignore it. ACP will re-remind after the cooldown expires.

**Memory filing reminders** -- After milestones (PR merged, issue closed, commit pushed), ACP waits a grace period for the agent to file memory unprompted. If it doesn't, ACP sends a nudge. If the agent does file memory within the grace window, the reminder is suppressed.

### Coming Soon

**Code word triggers** -- Agents embed code words in their workflow output (e.g., `[ACP:CHECKPOINT]`, `[ACP:REVIEW]`). The monitor detects these patterns in the JSONL and fires custom reminders. This lets agents schedule their own reminders without clock access.

**Task scheduling** -- Agent writes a structured entry like `[ACP:REMIND 5m "check test results"]` and the monitor delivers the reminder after the specified delay. Enables time-aware workflows for agents that cannot track wall-clock time.

**Custom trigger plugins** -- User-defined trigger classes loaded from config. Pattern match on any JSONL content -- tool outputs, error patterns, specific file changes.

**Multi-agent monitoring** -- Single ACP instance monitors multiple tmux sessions. Cross-agent awareness: "Agent A just compacted, remind Agent B to check shared state."

**Escalation chains** -- If a reminder is ignored N times, escalate: change wording, increase urgency, or notify a different channel.

**Metrics and dashboards** -- Track compaction frequency, context growth rate, reminder effectiveness over time. Export to JSON for visualization.

## Development

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run tests
python3 -m pytest tests/ -v

# 347 tests covering all modules:
#   test_cli.py            -- CLI commands, PID management, monitor loop
#   test_config.py         -- YAML loading, fallback parser, defaults
#   test_delivery.py       -- tmux delivery, idle detection, warmdown, audit trail
#   test_file_tracker.py   -- session discovery, rotation detection
#   test_token_monitor.py  -- token parsing, incremental reads, thresholds
#   test_compaction.py     -- trigger logic, grace/cooldown/stacking
#   test_memory_filing.py  -- milestone detection, pattern matching, lifecycle
```

PRs welcome. The codebase has zero external runtime dependencies by design -- keep it that way.

## License

MIT
