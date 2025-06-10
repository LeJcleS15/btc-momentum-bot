import { SignalCache } from "./signal";
import type { Signal, LiquidityCheck } from "./types";
import { log } from "./logger";
import { TRADING_CONFIG, RISK_CONFIG, config } from "./config";
import {
  DriftClient,
  Wallet,
  initialize,
  BulkAccountLoader,
  PerpMarkets,
  OrderType,
  PositionDirection,
  BASE_PRECISION,
  QUOTE_PRECISION,
  BN,
} from "@drift-labs/sdk";
import { Keypair, PublicKey } from "@solana/web3.js";
import { Connection } from "@solana/web3.js";

let driftClient: DriftClient;
let btcMarketIndex: number;

enum BotState {
  INITIALIZING = "INITIALIZING",
  HEALTHY = "HEALTHY",
  EMERGENCY = "EMERGENCY",
  SHUTDOWN = "SHUTDOWN",
}

export class MomentumBot {
  private state: BotState = BotState.INITIALIZING;
  private currentPosition: Signal = 0;
  private cycleCount = 0;
  private isCycleRunning = false;
  private cycleInterval: NodeJS.Timeout | null = null;
  private signalCache = new SignalCache();

  async initialize(): Promise<{ success: boolean; error?: string }> {
    try {
      log.cycle(0, "Bot initialization started");

      const driftInit = await this.initializeDrift();
      if (!driftInit.success) return driftInit;

      await this.reconstructPosition();

      this.state = BotState.HEALTHY;
      log.cycle(0, "Bot initialized successfully", { state: this.state });

      return { success: true };
    } catch (error) {
      this.state = BotState.EMERGENCY;
      return {
        success: false,
        error: `Bot initialization failed: ${(error as Error).message}`,
      };
    }
  }

  private async initializeDrift(): Promise<{
    success: boolean;
    error?: string;
  }> {
    try {
      log.cycle(0, "Initializing Drift client");

      const connection = new Connection(config.RPC_HTTP, {
        wsEndpoint: config.RPC_WS,
        commitment: "confirmed",
      });

      const sdk = initialize({ env: TRADING_CONFIG.ENV });
      const accountLoader = new BulkAccountLoader(connection, "confirmed", 1000);

      let secretKey: Uint8Array;
      try {
        secretKey = config.PRIVATE_KEY.startsWith("[")
          ? new Uint8Array(JSON.parse(config.PRIVATE_KEY))
          : Buffer.from(config.PRIVATE_KEY, "base64");
      } catch (error) {
        return {
          success: false,
          error: `Failed to parse private key: ${(error as Error).message}`,
        };
      }

      const wallet = new Wallet(Keypair.fromSecretKey(secretKey));
      log.cycle(0, "Wallet loaded", { address: wallet.publicKey.toString() });

      driftClient = new DriftClient({
        connection,
        wallet,
        programID: new PublicKey(sdk.DRIFT_PROGRAM_ID),
        authority: new PublicKey(config.AUTHORITY_KEY),
        subAccountIds: [5],
        activeSubAccountId: 5,
        accountSubscription: {
          type: "websocket",
          //@ts-ignore
          accountLoader,
        },
      });

      await driftClient.subscribe();
      const user = driftClient.getUser();
      await user.exists();

      const btcMarket = PerpMarkets[TRADING_CONFIG.ENV].find((market) => market.baseAssetSymbol === "BTC");

      if (!btcMarket) {
        return {
          success: false,
          error: "BTC market not found",
        };
      }

      btcMarketIndex = btcMarket.marketIndex;
      log.cycle(0, "Drift client initialized", {
        btcMarketIndex,
      });
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: `Drift initialization failed: ${(error as Error).message}`,
      };
    }
  }

  private async checkLiquidity(side: "bids" | "asks", orderSize: number): Promise<LiquidityCheck> {
    try {
      const response = await fetch(`https://dlob.drift.trade/l2?marketName=BTC-PERP`);
      if (!response.ok) throw new Error(`DLOB API error: ${response.status}`);

      const data = (await response.json()) as any;
      const orders = data[side];

      if (!orders || orders.length === 0) {
        return { canFill: false, estimatedSlippage: Infinity };
      }

      let totalSize = 0;
      let totalValue = 0;
      const bestPrice = parseFloat(orders[0].price) / QUOTE_PRECISION.toNumber();

      for (const order of orders) {
        const levelPrice = parseFloat(order.price) / QUOTE_PRECISION.toNumber();
        const levelSize = parseFloat(order.size) / BASE_PRECISION.toNumber();

        const fillSize = Math.min(levelSize, orderSize - totalSize);
        totalValue += fillSize * levelPrice;
        totalSize += fillSize;

        if (totalSize >= orderSize) break;
      }

      if (totalSize < orderSize) {
        return { canFill: false, estimatedSlippage: Infinity };
      }

      const avgPrice = totalValue / totalSize;
      const slippage = Math.abs((avgPrice - bestPrice) / bestPrice);

      return { canFill: true, estimatedSlippage: slippage };
    } catch (error) {
      log.error("LIQUIDITY", `Failed to check BTC ${side}`, error as Error);
      return { canFill: false, estimatedSlippage: Infinity };
    }
  }

  private validateSlippage(slippage: number, isClosing: boolean = false): { success: boolean; error?: string } {
    const maxSlippage = isClosing ? RISK_CONFIG.CLOSE_MAX_SLIPPAGE_BPS : RISK_CONFIG.MAX_SLIPPAGE_BPS;
    const maxSlippageDecimal = maxSlippage / 10000;

    if (slippage > maxSlippageDecimal) {
      return {
        success: false,
        error: `Slippage too high: ${(slippage * 100).toFixed(3)}% > ${(maxSlippageDecimal * 100).toFixed(3)}%`,
      };
    }
    return { success: true };
  }

  private validateLiquidity(
    liquidityCheck: LiquidityCheck,
    isClosing: boolean = false
  ): { success: boolean; error?: string } {
    if (!liquidityCheck.canFill) {
      return {
        success: false,
        error: "Insufficient market liquidity",
      };
    }

    return this.validateSlippage(liquidityCheck.estimatedSlippage, isClosing);
  }

  private async reconstructPosition(): Promise<void> {
    try {
      const positions = driftClient.getUser().getUserAccount().perpPositions;

      for (const pos of positions) {
        if (pos.marketIndex === btcMarketIndex && !pos.baseAssetAmount.eq(new BN(0))) {
          const size = pos.baseAssetAmount.toNumber() / BASE_PRECISION.toNumber();
          this.currentPosition = size > 0 ? 1 : -1;
          log.cycle(0, "Reconstructed BTC position", {
            position: this.getSignalString(this.currentPosition),
            size,
          });
          return;
        }
      }

      this.currentPosition = 0;
      log.cycle(0, "No existing BTC position found");
    } catch (error) {
      log.error("POSITION", "Failed to reconstruct position", error as Error);
      this.currentPosition = 0;
    }
  }

  start(): void {
    if (this.state !== BotState.HEALTHY) {
      log.error("BOT", "Cannot start bot in non-healthy state", new Error(`Current state: ${this.state}`));
      return;
    }

    log.cycle(0, "Starting momentum trading");

    this.executeCycle();

    this.cycleInterval = setInterval(() => {
      this.executeCycle();
    }, TRADING_CONFIG.CYCLE_INTERVAL_MS);
  }

  async stop(): Promise<void> {
    log.cycle(0, "Stopping bot");

    if (this.cycleInterval) {
      clearInterval(this.cycleInterval);
      this.cycleInterval = null;
    }

    while (this.isCycleRunning) {
      log.cycle(0, "Waiting for current cycle to complete");
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    if (this.currentPosition !== 0) {
      log.cycle(0, "Closing position before shutdown");
      await this.closePosition();
    }

    if (driftClient) {
      await driftClient.unsubscribe();
      log.cycle(0, "Drift client disconnected");
    }

    log.cycle(0, "Bot stopped", { totalCycles: this.cycleCount });
  }

  private async executeCycle(): Promise<void> {
    if (this.isCycleRunning) {
      log.cycle(this.cycleCount + 1, "Previous cycle still running, skipping");
      return;
    }

    this.isCycleRunning = true;
    this.cycleCount++;
    const cycleStart = process.hrtime.bigint();

    try {
      log.cycle(this.cycleCount, "Cycle started");

      if (this.state === BotState.EMERGENCY) {
        log.cycle(this.cycleCount, "Bot in emergency state, closing positions");
        await this.closePosition();
        return;
      }

      const { signal, age } = await this.signalCache.getSignal();

      log.cycle(this.cycleCount, "Signal retrieved", {
        signal: this.getSignalString(signal),
        age: Math.round(age / 1000),
        currentPosition: this.getSignalString(this.currentPosition),
      });

      if (signal !== this.currentPosition) {
        await this.changePosition(signal);
      } else {
        log.cycle(this.cycleCount, "Holding current position", {
          position: this.getSignalString(this.currentPosition),
        });
      }

      this.state = BotState.HEALTHY;
    } catch (error) {
      log.error("CYCLE", `Cycle ${this.cycleCount} failed`, error as Error);
      this.state = BotState.EMERGENCY;
    } finally {
      const cycleTime = Number(process.hrtime.bigint() - cycleStart) / 1e6;
      log.cycle(this.cycleCount, "Cycle completed", {
        duration: `${cycleTime.toFixed(1)}ms`,
      });
      this.isCycleRunning = false;
    }
  }

  private async changePosition(newSignal: Signal): Promise<void> {
    const currentStr = this.getSignalString(this.currentPosition);
    const newStr = this.getSignalString(newSignal);

    log.cycle(this.cycleCount, `Position transition: ${currentStr} -> ${newStr}`);

    try {
      if (this.currentPosition !== 0) {
        await this.closePosition();
      }

      if (newSignal !== 0) {
        await this.openPosition(newSignal);
      }

      this.currentPosition = newSignal;
      log.position(`Position successfully changed to ${newStr}`);
    } catch (error) {
      log.error("TRANSITION", "Position transition failed", error as Error);
      this.state = BotState.EMERGENCY;
      throw error;
    }
  }

  private async openPosition(signal: Signal): Promise<void> {
    const direction = signal > 0 ? PositionDirection.LONG : PositionDirection.SHORT;
    const directionStr = signal > 0 ? "LONG" : "SHORT";
    const side = signal > 0 ? "asks" : "bids";

    log.position(`Opening ${directionStr} BTC position`, {
      quantity: TRADING_CONFIG.BTC_QUANTITY,
    });

    const liquidityCheck = await this.checkLiquidity(side, TRADING_CONFIG.BTC_QUANTITY);
    const validation = this.validateLiquidity(liquidityCheck, false);

    if (!validation.success) {
      log.error("POSITION", "Liquidity validation failed", new Error(validation.error));
      return;
    }

    log.position("Liquidity check passed", {
      side,
      slippage: `${(liquidityCheck.estimatedSlippage * 100).toFixed(3)}%`,
    });

    for (let attempt = 1; attempt <= RISK_CONFIG.MAX_RETRIES; attempt++) {
      try {
        const tx = await driftClient.placePerpOrder({
          orderType: OrderType.MARKET,
          marketIndex: btcMarketIndex,
          direction,
          baseAssetAmount: new BN(TRADING_CONFIG.BTC_QUANTITY * BASE_PRECISION.toNumber()),
          reduceOnly: false,
        });

        log.order("OPEN", "BTC-PERP", `${directionStr} position opened`, {
          txSignature: tx,
          quantity: TRADING_CONFIG.BTC_QUANTITY,
          marketIndex: btcMarketIndex,
          slippage: `${(liquidityCheck.estimatedSlippage * 100).toFixed(3)}%`,
        });

        return;
      } catch (error) {
        if (attempt === RISK_CONFIG.MAX_RETRIES) {
          throw new Error(`Order placement failed after ${attempt} attempts: ${(error as Error).message}`);
        }

        const delay = RISK_CONFIG.RETRY_DELAY_MS * Math.pow(2, attempt - 1);
        log.error("ORDER", `Order attempt ${attempt} failed, retrying in ${delay}ms`, error as Error);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  private async closePosition(): Promise<void> {
    if (this.currentPosition === 0) {
      log.position("No position to close");
      return;
    }

    const currentStr = this.getSignalString(this.currentPosition);
    const direction = this.currentPosition > 0 ? PositionDirection.SHORT : PositionDirection.LONG;
    const side = this.currentPosition > 0 ? "bids" : "asks";

    log.position(`Closing ${currentStr} position`);

    const liquidityCheck = await this.checkLiquidity(side, TRADING_CONFIG.BTC_QUANTITY);
    const validation = this.validateLiquidity(liquidityCheck, true);

    if (!validation.success) {
      log.risk("Close liquidity validation failed, forcing close anyway", {
        error: validation.error,
        slippage: `${(liquidityCheck.estimatedSlippage * 100).toFixed(3)}%`,
      });
    } else {
      log.position("Close liquidity check passed", {
        side,
        slippage: `${(liquidityCheck.estimatedSlippage * 100).toFixed(3)}%`,
      });
    }

    for (let attempt = 1; attempt <= RISK_CONFIG.MAX_RETRIES; attempt++) {
      try {
        const tx = await driftClient.placePerpOrder({
          orderType: OrderType.MARKET,
          marketIndex: btcMarketIndex,
          direction,
          baseAssetAmount: new BN(TRADING_CONFIG.BTC_QUANTITY * BASE_PRECISION.toNumber()),
          reduceOnly: true,
        });

        log.order("CLOSE", "BTC-PERP", `${currentStr} position closed`, {
          txSignature: tx,
          quantity: TRADING_CONFIG.BTC_QUANTITY,
          marketIndex: btcMarketIndex,
          slippage: `${(liquidityCheck.estimatedSlippage * 100).toFixed(3)}%`,
        });

        this.currentPosition = 0;
        return;
      } catch (error) {
        if (attempt === RISK_CONFIG.MAX_RETRIES) {
          throw new Error(`Close order failed after ${attempt} attempts: ${(error as Error).message}`);
        }

        const delay = RISK_CONFIG.RETRY_DELAY_MS * Math.pow(2, attempt - 1);
        log.error("ORDER", `Close attempt ${attempt} failed, retrying in ${delay}ms`, error as Error);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  private getSignalString(signal: Signal): string {
    return signal === 1 ? "LONG" : signal === -1 ? "SHORT" : "FLAT";
  }
}
