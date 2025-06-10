export type CandleData = {
  ts: number;
  fillHigh: number;
  fillOpen: number;
  fillClose: number;
  fillLow: number;
  oracleOpen: number;
  oracleHigh: number;
  oracleClose: number;
  oracleLow: number;
  quoteVolume: number;
  baseVolume: number;
};

export type Signal = -1 | 0 | 1;

export type LiquidityCheck = {
  canFill: boolean;
  estimatedSlippage: number;
};
