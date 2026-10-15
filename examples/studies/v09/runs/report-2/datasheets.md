# v0.9 datasheets

Public archives used for non-commercial research under the providers' terms; raw data are never committed or bundled. Capabilities are those of the public feeds (no order ids, no queue positions, no trader identities).

## Pool `dev` (development)

- Period: 2023-07-15 to 2023-07-15; primaries per stream: 1; backups: 0
- Streams: bybit-linear:BTCUSDT, bybit-spot:BTCUSDT
- Declared / accessed / NOT_AVAILABLE attempts: 2 / 2 / 0

## Pool `explore` (explore)

- Period: 2023-08-01 to 2023-10-31; primaries per stream: 8; backups: 2
- Streams: bybit-linear:BTCUSDT, bybit-linear:ETHUSDT, binance-um:BTCUSDT, binance-um:ETHUSDT
- Declared / accessed / NOT_AVAILABLE attempts: 40 / 34 / 2

## Pool `confirm` (confirm)

- Period: 2023-11-01 to 2024-03-28; primaries per stream: 40; backups: 6
- Streams: bybit-linear:BTCUSDT, bybit-linear:ETHUSDT, binance-um:BTCUSDT, binance-um:ETHUSDT
- Declared / accessed / NOT_AVAILABLE attempts: 184 / 35 / 6

## Pool `confirm_late` (confirm_late)

- Period: 2024-06-01 to 2024-12-31; primaries per stream: 20; backups: 4
- Streams: bybit-linear:BTCUSDT, bybit-linear:ETHUSDT
- Declared / accessed / NOT_AVAILABLE attempts: 48 / 17 / 2

## Pool `final` (final)

- Period: 2025-05-01 to 2025-08-28; primaries per stream: 12; backups: 3
- Streams: bybit-linear:SOLUSDT, bybit-linear:BTCUSDT, bybit-spot:BTCUSDT
- Declared / accessed / NOT_AVAILABLE attempts: 45 / 36 / 4

### bybit-linear BTCUSDT

- Provider: bybit public data archive; types: l2_snapshot_delta, trades; tick 0.1, lot 0.001
- Capabilities: depth 500, incremental L2 True, aggressor side True, order ids False, queue clearance observable True

### bybit-linear ETHUSDT

- Provider: bybit public data archive; types: l2_snapshot_delta, trades; tick 0.01, lot 0.01
- Capabilities: depth 500, incremental L2 True, aggressor side True, order ids False, queue clearance observable True

### binance-um BTCUSDT

- Provider: binance public data archive; types: l1_bookticker, trades; tick 0.1, lot 0.001
- Capabilities: depth 1, incremental L2 False, aggressor side True, order ids False, queue clearance observable False

### binance-um ETHUSDT

- Provider: binance public data archive; types: l1_bookticker, trades; tick 0.01, lot 0.001
- Capabilities: depth 1, incremental L2 False, aggressor side True, order ids False, queue clearance observable False

### bybit-linear SOLUSDT

- Provider: bybit public data archive; types: l2_snapshot_delta, trades; tick 0.01, lot 0.1
- Capabilities: depth 500, incremental L2 True, aggressor side True, order ids False, queue clearance observable True

### bybit-spot BTCUSDT

- Provider: bybit public data archive; types: l2_snapshot_delta, trades; tick 0.01, lot 1e-06
- Capabilities: depth 200, incremental L2 True, aggressor side True, order ids False, queue clearance observable True

