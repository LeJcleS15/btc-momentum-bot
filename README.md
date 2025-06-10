# BTC Momentum Bot

Automated momentum trading bot for BTC-PERP on Drift Protocol using ensemble EMA strategies.

## Strategy

- 5 EMA pairs with median signal aggregation
- Volume filtering (20-period average threshold)
- 2-bar execution lag for signal stability
- Fixed position size: 0.01 BTC
