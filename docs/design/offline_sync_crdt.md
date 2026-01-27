# Offline-First Sync Design (CRDTs)

## Problem
Tuners often work in garages or trackside with poor/no internet. They need to view logs, edit maps, and save notes offline, then sync when connected.

## Solution: Convergent Replicated Data Types (CRDTs)

We will use a **State-based CRDT** approach for merging calibration changes and logs.

### Architecture
1.  **Local DB**: SQLite (Desktop) / PouchDB (Web) stores local state.
2.  **Remote DB**: PostgreSQL with a sync layer (e.g., ElectricSQL or custom sync endpoint).

### Data Structures

#### Calibration Tables (Last-Write-Wins Map)
For a fuel table `MainFuel[RPM][Load]`:
- Each cell is a separate register with a timestamp.
- `LWW-Register`: `{ value: 14.7, timestamp: 1715000000, clientId: 'device_A' }`
- Merge function:
  ```typescript
  if (remote.timestamp > local.timestamp) updated = remote
  else if (remote.timestamp == local.timestamp && remote.clientId > local.clientId) updated = remote
  ```

#### Logs (Append-Only Set)
- Logs are immutable time-series data.
- Sync logic: simply download any logs with `timestamp > last_sync_timestamp`. No merge conflicts possible.

### Conflict Resolution
- **Automatic**: For cell-level edits, LWW (Last Write Wins) based on wall-clock time is sufficient for most tuning scenarios.
- **Manual**: If a major version conflict occurs (e.g., two completely different base maps loaded), prompt user to choose "Local" or "Server" copy.
