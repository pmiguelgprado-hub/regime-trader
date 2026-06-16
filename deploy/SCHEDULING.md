# Scheduling & reliability — "se haga siempre"

How the bot stays running unattended, and what to do so the daily jobs fire **on time**.

## The jobs (launchd LaunchAgents)

All installed in `~/Library/LaunchAgents/` (persist across reboot/login) and loaded.
Verify with `launchctl list | grep regimetrader` (col 2 = last exit code; 0 = clean).

| Job | When (weekdays) | Does |
|---|---|---|
| `com.regimetrader.shadowlog` | 22:15 | Shadow study (read-only): JM/BOCPD regime + VIX/FRED macro + refit-equivalence |
| `com.regimetrader.rebalance` | 22:30 | Cross-sectional book rebalance (`--execute`, PAPER) |
| `com.regimetrader.quality` | ~22:50 | Quality sleeve (dry-run + synthetic NAV) |
| `com.regimetrader.challenger` | — | Challenger book (dry-run) |
| `com.regimetrader.recordtrack` | 23:00 | Append the day's NAV row to `track_record.csv` (gate evidence) |
| `com.regimetrader.riskcheck` | every 15 min | Intraday risk ladder + **heartbeat** (T0.5) |
| `com.regimetrader.runonce` | daily | Single decision cycle (SPY baseline) |

`RunAtLoad=false` for all: on login launchd loads the agent but runs it only at the
calendar time (a rebalance must not fire on every login).

## Three-layer reliability model

1. **On-time layer — scheduled wake.** macOS launchd does NOT wake a sleeping Mac for a
   `StartCalendarInterval` job. On a laptop that sleeps at 22:15/22:30 the run happens
   late (see layer 2). To run on time, schedule a power-management wake just before the
   first job:

   ```sh
   sudo pmset repeat wakeorpoweron MTWRF 22:10:00   # wake weekdays 22:10
   pmset -g sched                                   # verify the repeating wake is listed
   ```

   **Only reliable on AC power.** On battery with the lid closed macOS may refuse to wake.
   Keep the Mac plugged in overnight. To remove: `sudo pmset repeat cancel`.

2. **Self-heal layer — launchd catch-up.** If the Mac was asleep/off at the scheduled
   time, launchd runs the missed calendar job **once on the next wake**. Proven 2026-06-15:
   the Mac was asleep at 22:15/22:30 and the jobs ran at 22:32 on wake, writing all rows.
   Late runs are safe: the open-order guard makes a same-day re-run a no-op, and the daily
   cadence just re-scales gross to current vol.

3. **Detective layer — heartbeat alarm.** `--risk-check` checks `track_record.csv`
   freshness; if the last row is >2 business days old it sends a **Telegram CRITICAL**
   (deduplicated per day). So a genuinely missed run gets surfaced, not silently dropped.

## What software cannot fix

A laptop that is **powered off** (not asleep) at the scheduled time, and stays off,
cannot run that day's job until it is next powered on. True 24/7 on-time execution needs
an always-on host (Mac mini on AC, or a VPS with the launchd jobs ported to cron/systemd).

## Quick health check

```sh
launchctl list | grep regimetrader                 # all loaded, exit 0
tail -1 track_record.csv                            # today's NAV row present
tail -1 logs/shadow_regime.csv                      # today's shadow row present
pmset -g sched                                      # scheduled wake present (layer 1)
pmset -g batt                                       # on AC for reliable overnight wake
```
