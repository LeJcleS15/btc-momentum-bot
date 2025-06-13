import { SignalCache } from './signal';
import type { Signal, LiquidityCheck, PerformanceSnapshot } from './types';
import { log } from './logger';
import { TRADING_CONFIG, RISK_CONFIG, config, SWIFT_CONFIG } from './config';
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
	MarketType,
	type PerpMarketAccount,
	type OrderParams,
	getMarketOrderParams,
	getSignedMsgUserAccountPublicKey,
} from '@drift-labs/sdk';
import { Keypair, PublicKey } from '@solana/web3.js';
import { Connection } from '@solana/web3.js';

let driftClient: DriftClient;
let btcMarketIndex: number;

// Cache frequently used values
const BASE_PRECISION_NUM = BASE_PRECISION.toNumber();
const QUOTE_PRECISION_NUM = QUOTE_PRECISION.toNumber();
const TRADING_QUANTITY_SCALED =
	TRADING_CONFIG.BTC_QUANTITY * BASE_PRECISION_NUM;
const MAX_SLIPPAGE_DECIMAL = RISK_CONFIG.MAX_SLIPPAGE_BPS / 10000;
const CLOSE_MAX_SLIPPAGE_DECIMAL = RISK_CONFIG.CLOSE_MAX_SLIPPAGE_BPS / 10000;

// Pre-computed signal strings
const SIGNAL_STRINGS = ['SHORT', 'FLAT', 'LONG'] as const;
const getSignalString = (signal: Signal): string => SIGNAL_STRINGS[signal + 1]!;

enum BotState {
	INITIALIZING = 'INITIALIZING',
	HEALTHY = 'HEALTHY',
	EMERGENCY = 'EMERGENCY',
	SHUTDOWN = 'SHUTDOWN',
}

export class MomentumBot {
	private state: BotState = BotState.INITIALIZING;
	private currentPosition: Signal = 0;
	private cycleCount = 0;
	private isCycleRunning = false;
	private cycleInterval: NodeJS.Timeout | null = null;
	private signalCache = new SignalCache();

	private strategyStartingEquity: number = 0;
	private strategyRealizedPnl: number = 0;

	private strategyTotalFunding: number = 0;
	private lastFundingPnl: number = 0;

	async initialize(): Promise<{ success: boolean; error?: string }> {
		try {
			log.cycle(0, 'Bot initialization started');

			const driftInit = await this.initializeDrift();
			if (!driftInit.success) return driftInit;

			log.cycle(0, 'Configuration loaded', {
				btcQuantity: TRADING_CONFIG.BTC_QUANTITY,
				cycleInterval: `${TRADING_CONFIG.CYCLE_INTERVAL_MS / 1000}s`,
				env: TRADING_CONFIG.ENV,
			});

			const user = driftClient.getUser();
			this.strategyStartingEquity =
				user.getTotalCollateral().toNumber() / QUOTE_PRECISION_NUM;

			this.lastFundingPnl =
				user.getUnrealizedFundingPNL().toNumber() / QUOTE_PRECISION_NUM;

			log.cycle(0, 'Strategy tracking initialized', {
				startingEquity: this.strategyStartingEquity,
			});

			this.reconstructPosition();

			this.state = BotState.HEALTHY;
			log.cycle(0, 'Bot initialized successfully', { state: this.state });

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
			log.cycle(0, 'Initializing Drift client');

			const connection = new Connection(config.RPC_HTTP, {
				wsEndpoint: config.RPC_WS,
				commitment: 'confirmed',
			});

			const sdk = initialize({ env: TRADING_CONFIG.ENV });
			const accountLoader = new BulkAccountLoader(
				connection,
				'confirmed',
				1000
			);

			let secretKey: Uint8Array;
			try {
				secretKey = config.PRIVATE_KEY.startsWith('[')
					? new Uint8Array(JSON.parse(config.PRIVATE_KEY))
					: Buffer.from(config.PRIVATE_KEY, 'base64');
			} catch (error) {
				return {
					success: false,
					error: `Failed to parse private key: ${(error as Error).message}`,
				};
			}

			const wallet = new Wallet(Keypair.fromSecretKey(secretKey));
			log.cycle(0, 'Wallet loaded', { address: wallet.publicKey.toString() });

			driftClient = new DriftClient({
				connection,
				wallet,
				programID: new PublicKey(sdk.DRIFT_PROGRAM_ID),
				authority: new PublicKey(config.AUTHORITY_KEY),
				subAccountIds: [5],
				activeSubAccountId: 5,
				accountSubscription: {
					type: 'websocket',
					//@ts-ignore
					accountLoader,
				},
			});

			await driftClient.subscribe();
			const user = driftClient.getUser();
			await user.exists();

			// const signedMsgUserAccount = getSignedMsgUserAccountPublicKey(
			// 	driftClient.program.programId,
			// 	new PublicKey(config.AUTHORITY_KEY)
			// );

			// const acc =
			// 	await driftClient.connection.getAccountInfo(signedMsgUserAccount);
			// if (acc === null) {
			// 	log.cycle(
			// 		0,
			// 		`Creating SignedMsgUserAccount for ${config.AUTHORITY_KEY}`
			// 	);
			// 	const [txSig, signedMsgUserAccountPublicKey] =
			// 		await driftClient.initializeSignedMsgUserOrders(
			// 			new PublicKey(config.AUTHORITY_KEY),
			// 			8
			// 		);
			// 	log.cycle(0, 'SignedMsgUserAccount created', {
			// 		account: signedMsgUserAccountPublicKey.toBase58(),
			// 		txSignature: txSig,
			// 	});
			// } else {
			// 	log.cycle(0, 'SignedMsgUserAccount already exists', {
			// 		account: signedMsgUserAccount.toBase58(),
			// 	});
			// }

			const btcMarket = PerpMarkets[TRADING_CONFIG.ENV].find(
				(market) => market.baseAssetSymbol === 'BTC'
			);

			if (!btcMarket) {
				return {
					success: false,
					error: 'BTC market not found',
				};
			}

			btcMarketIndex = btcMarket.marketIndex;
			log.cycle(0, 'Drift client initialized', {
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

	private getCurrentBTCPosition(): { signal: Signal; size: number } {
		try {
			const perpPosition = driftClient
				.getUser()
				.getPerpPosition(btcMarketIndex);

			if (!perpPosition || perpPosition.baseAssetAmount.eq(new BN(0))) {
				return { signal: 0, size: 0 };
			}

			const size = perpPosition.baseAssetAmount.toNumber() / BASE_PRECISION_NUM;
			return {
				signal: size > 0 ? 1 : -1,
				size: size > 0 ? size : -size,
			};
		} catch (error) {
			log.error('POSITION', 'Failed to get current position', error as Error);
			return { signal: 0, size: 0 };
		}
	}

	private async checkLiquidity(
		side: 'bids' | 'asks',
		orderSize: number
	): Promise<LiquidityCheck> {
		try {
			const response = await fetch(
				`https://dlob.drift.trade/l2?marketName=BTC-PERP`
			);
			if (!response.ok) throw new Error(`DLOB API error: ${response.status}`);

			const data = (await response.json()) as any;
			const orders = data[side];

			if (!orders || orders.length === 0) {
				return { canFill: false, estimatedSlippage: Infinity };
			}

			let totalSize = 0;
			let totalValue = 0;
			const bestPrice = +orders[0].price / QUOTE_PRECISION_NUM;

			for (let i = 0; i < orders.length; i++) {
				const order = orders[i];
				const levelPrice = +order.price / QUOTE_PRECISION_NUM;
				const levelSize = +order.size / BASE_PRECISION_NUM;

				const remaining = orderSize - totalSize;
				if (remaining <= 0) break;

				const fillSize = levelSize < remaining ? levelSize : remaining;
				totalValue += fillSize * levelPrice;
				totalSize += fillSize;

				if (totalSize >= orderSize) break;
			}

			if (totalSize < orderSize) {
				return { canFill: false, estimatedSlippage: Infinity };
			}

			const avgPrice = totalValue / totalSize;
			const slippage =
				avgPrice > bestPrice
					? (avgPrice - bestPrice) / bestPrice
					: (bestPrice - avgPrice) / bestPrice;

			return { canFill: true, estimatedSlippage: slippage };
		} catch (error) {
			log.error('LIQUIDITY', `Failed to check BTC ${side}`, error as Error);
			return { canFill: false, estimatedSlippage: Infinity };
		}
	}

	private validateLiquidity(
		liquidityCheck: LiquidityCheck,
		isClosing: boolean = false
	): { success: boolean; error?: string; slippagePercent: number } {
		if (!liquidityCheck.canFill) {
			return {
				success: false,
				error: 'Insufficient market liquidity',
				slippagePercent: 0,
			};
		}

		const maxSlippage = isClosing
			? CLOSE_MAX_SLIPPAGE_DECIMAL
			: MAX_SLIPPAGE_DECIMAL;
		const slippagePercent = liquidityCheck.estimatedSlippage * 100;

		if (liquidityCheck.estimatedSlippage > maxSlippage) {
			return {
				success: false,
				error: `Slippage too high: ${slippagePercent.toFixed(3)}% > ${(maxSlippage * 100).toFixed(3)}%`,
				slippagePercent,
			};
		}
		return { success: true, slippagePercent };
	}

	private reconstructPosition(): void {
		const position = this.getCurrentBTCPosition();
		this.currentPosition = position.signal;

		if (position.signal !== 0) {
			log.cycle(0, 'Reconstructed BTC position', {
				position: getSignalString(position.signal),
				size: position.size,
			});
		} else {
			log.cycle(0, 'No existing BTC position found');
		}
	}

	start(): void {
		if (this.state !== BotState.HEALTHY) {
			log.error(
				'BOT',
				'Cannot start bot in non-healthy state',
				new Error(`Current state: ${this.state}`)
			);
			return;
		}

		log.cycle(0, 'Starting momentum trading');
		this.executeCycle();
		this.cycleInterval = setInterval(
			() => this.executeCycle(),
			TRADING_CONFIG.CYCLE_INTERVAL_MS
		);
	}

	async stop(): Promise<void> {
		log.cycle(0, 'Stopping bot');

		if (this.cycleInterval) {
			clearInterval(this.cycleInterval);
			this.cycleInterval = null;
		}

		while (this.isCycleRunning) {
			log.cycle(0, 'Waiting for current cycle to complete');
			await new Promise((resolve) => setTimeout(resolve, 1000));
		}

		if (this.currentPosition !== 0) {
			log.cycle(0, 'Closing position before shutdown');
			await this.closePosition();
		}

		if (driftClient) {
			await driftClient.unsubscribe();
			log.cycle(0, 'Drift client disconnected');
		}

		log.cycle(0, 'Bot stopped', { totalCycles: this.cycleCount });
	}

	private async executeCycle(): Promise<void> {
		if (this.isCycleRunning) {
			log.cycle(this.cycleCount + 1, 'Previous cycle still running, skipping');
			return;
		}

		this.isCycleRunning = true;
		this.cycleCount++;
		const cycleStart = process.hrtime.bigint();

		try {
			log.cycle(this.cycleCount, 'Cycle started');

			if (this.state === BotState.EMERGENCY) {
				log.cycle(this.cycleCount, 'Bot in emergency state, closing positions');
				await this.closePosition();
				return;
			}

			const signalData = await this.signalCache.getSignal();
			const currentStr = getSignalString(this.currentPosition);
			const signalStr = getSignalString(signalData.signal);

			log.cycle(this.cycleCount, 'Signal retrieved', {
				signal: signalStr,
				age: Math.round(signalData.age / 1000),
				currentPosition: currentStr,
			});

			if (signalData.signal !== this.currentPosition) {
				await this.changePosition(signalData.signal, currentStr, signalStr);
			} else {
				log.cycle(this.cycleCount, 'Holding current position', {
					position: currentStr,
				});
			}

			this.state = BotState.HEALTHY;
		} catch (error) {
			log.error('CYCLE', `Cycle ${this.cycleCount} failed`, error as Error);
			this.state = BotState.EMERGENCY;
		} finally {
			const snapshot = await this.capturePerformanceSnapshot();
			if (snapshot) {
				log.snapshot(snapshot);
			}
			const cycleTime = Number(process.hrtime.bigint() - cycleStart) / 1e6;
			log.cycle(this.cycleCount, 'Cycle completed', {
				duration: `${cycleTime.toFixed(1)}ms`,
			});
			this.isCycleRunning = false;
		}
	}

	private async changePosition(
		newSignal: Signal,
		currentStr: string,
		newStr: string
	): Promise<void> {
		log.cycle(
			this.cycleCount,
			`Position transition: ${currentStr} -> ${newStr}`
		);

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
			log.error('TRANSITION', 'Position transition failed', error as Error);
			this.state = BotState.EMERGENCY;
			throw error;
		}
	}

	private async executeSwiftOrder(
		direction: PositionDirection,
		baseAssetAmount: BN
	): Promise<string> {
		try {
			const oraclePrice =
				driftClient.getOracleDataForPerpMarket(btcMarketIndex).price;

			const rangeBps = SWIFT_CONFIG.AUCTION_RANGE_BPS;
			const highPrice = oraclePrice.muln(10000 + rangeBps).divn(10000);
			const lowPrice = oraclePrice.muln(10000 - rangeBps).divn(10000);

			const isLong = direction === PositionDirection.LONG;

			const makerOrderParams = getMarketOrderParams({
				marketIndex: btcMarketIndex,
				marketType: MarketType.PERP,
				direction: direction,
				baseAssetAmount: baseAssetAmount,
				auctionStartPrice: isLong ? lowPrice : highPrice,
				auctionEndPrice: isLong ? highPrice : lowPrice,
				auctionDuration: SWIFT_CONFIG.AUCTION_DURATION_SLOTS,
			});

			const uuid = new Uint8Array(8);
			const now = Date.now();
			for (let i = 0; i < 4; i++) {
				uuid[i] = (now >>> (8 * (3 - i))) & 0xff;
			}

			crypto.getRandomValues(uuid.subarray(4));

			const orderMessage = {
				signedMsgOrderParams: makerOrderParams as OrderParams,
				// subAccountId: 5,
				slot: new BN(await driftClient.connection.getSlot()),
				uuid,
				stopLossOrderParams: null,
				takeProfitOrderParams: null,
				takerPubkey: new PublicKey(
					'CHyy6rPYPN79Ka5E8dxy8ANpUzs4qkQXaTJguUfBP3SN'
				),
			};

			const { orderParams: message, signature } =
				driftClient.signSignedMsgOrderParamsMessage(orderMessage, true);

			const response = await fetch(SWIFT_CONFIG.API_URL, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					market_index: btcMarketIndex,
					market_type: 'perp',
					message: message.toString(),
					signature: signature.toString('base64'),
					taker_authority: driftClient.authority.toBase58(),
					signing_authority: driftClient.wallet.publicKey.toBase58(),
				}),
				signal: AbortSignal.timeout(SWIFT_CONFIG.TIMEOUT_MS),
			});

			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Swift API ${response.status}: ${errorText}`);
			}

			const result = (await response.json()) as any;
			const txSignature = result.transaction_signature;

			if (!txSignature || txSignature.length < 87) {
				throw new Error(`Invalid swift signature length: ${txSignature}`);
			}

			// verify with exchange (30 second timeout)
			if (SWIFT_CONFIG.VERIFY_WITH_EXCHANGE) {
				const maxWaitTime = 30000; // 30 seconds
				const pollInterval = 2000; // 2 seconds
				const startTime = Date.now();

				while (Date.now() - startTime < maxWaitTime) {
					const confirmResponse = await fetch(
						`${SWIFT_CONFIG.API_URL}/confirmation/hash-status?hash=${encodeURIComponent(txSignature)}`,
						{ signal: AbortSignal.timeout(5000) }
					);

					if (confirmResponse.ok) {
						break; // Order confirmed/filled
					}

					await new Promise((resolve) => setTimeout(resolve, pollInterval));
				}
			}

			return txSignature;
		} catch (error) {
			throw new Error(`Swift failed: ${(error as Error).message}`);
		}
	}

	private async executeOrder(
		direction: PositionDirection,
		baseAssetAmount: BN,
		reduceOnly: boolean,
		orderType: string
	): Promise<string> {
		// Try Swift first (open)
		// if (SWIFT_CONFIG.ENABLED && !reduceOnly) {
		// 	try {
		// 		log.order('SWIFT', 'BTC-PERP', `Attempting Swift ${orderType}`, {
		// 			direction: direction === PositionDirection.LONG ? 'LONG' : 'SHORT',
		// 			size: baseAssetAmount.toNumber() / BASE_PRECISION_NUM,
		// 		});

		// 		const tx = await this.executeSwiftOrder(direction, baseAssetAmount);

		// 		log.order('SWIFT', 'BTC-PERP', `Swift ${orderType} successful`, {
		// 			txSignature: tx,
		// 		});
		// 		return tx;
		// 	} catch (error) {
		// 		log.risk('Swift order failed, falling back to market order', {
		// 			error: (error as Error).message,
		// 			orderType,
		// 		});
		// 	}
		// }

		// Fallback to market order with retries
		for (let attempt = 1; attempt <= RISK_CONFIG.MAX_RETRIES; attempt++) {
			try {
				const tx = await driftClient.placePerpOrder({
					orderType: OrderType.MARKET,
					marketIndex: btcMarketIndex,
					direction,
					baseAssetAmount,
					reduceOnly,
				});
				return tx;
			} catch (error) {
				if (attempt === RISK_CONFIG.MAX_RETRIES) {
					throw new Error(
						`${orderType} order failed after ${attempt} attempts: ${(error as Error).message}`
					);
				}

				const delay = RISK_CONFIG.RETRY_DELAY_MS * Math.pow(2, attempt - 1);
				log.error(
					'ORDER',
					`${orderType} attempt ${attempt} failed, retrying in ${delay}ms`,
					error as Error
				);
				await new Promise((resolve) => setTimeout(resolve, delay));
			}
		}
		throw new Error('Should not reach here');
	}

	private async openPosition(signal: Signal): Promise<void> {
		const direction =
			signal > 0 ? PositionDirection.LONG : PositionDirection.SHORT;
		const directionStr = signal > 0 ? 'LONG' : 'SHORT';
		const side = signal > 0 ? 'asks' : 'bids';

		log.position(`Opening ${directionStr} BTC position`, {
			quantity: TRADING_CONFIG.BTC_QUANTITY,
		});

		const liquidityCheck = await this.checkLiquidity(
			side,
			TRADING_CONFIG.BTC_QUANTITY
		);
		const validation = this.validateLiquidity(liquidityCheck, false);

		if (!validation.success) {
			log.error(
				'POSITION',
				'Liquidity validation failed',
				new Error(validation.error)
			);
			return;
		}

		log.position('Liquidity check passed', {
			side,
			slippage: `${validation.slippagePercent.toFixed(3)}%`,
		});

		const baseAssetAmount = new BN(TRADING_QUANTITY_SCALED);
		const tx = await this.executeOrder(
			direction,
			baseAssetAmount,
			false,
			'Open'
		);

		log.order('OPEN', 'BTC-PERP', `${directionStr} position opened`, {
			txSignature: tx,
			quantity: TRADING_CONFIG.BTC_QUANTITY,
			marketIndex: btcMarketIndex,
			slippage: `${validation.slippagePercent.toFixed(3)}%`,
		});
	}

	private calculatePositionPnl(
		signal: Signal,
		size: number,
		entryPrice: number,
		currentPrice: number
	): number {
		return signal * size * (currentPrice - entryPrice);
	}

	private getPositionEntryPrice(): number {
		const perpPosition = driftClient.getUser().getPerpPosition(btcMarketIndex);
		if (!perpPosition || perpPosition.baseAssetAmount.eq(new BN(0))) return 0;

		return (
			perpPosition.quoteEntryAmount.abs().toNumber() /
			QUOTE_PRECISION_NUM /
			(perpPosition.baseAssetAmount.abs().toNumber() / BASE_PRECISION_NUM)
		);
	}

	private async closePosition(): Promise<void> {
		const position = this.getCurrentBTCPosition();

		if (position.signal === 0) {
			log.position('No position to close');
			this.currentPosition = 0;
			return;
		}

		const entryPrice = this.getPositionEntryPrice();
		const currentStr = getSignalString(position.signal);
		const direction =
			position.signal > 0 ? PositionDirection.SHORT : PositionDirection.LONG;
		const side = position.signal > 0 ? 'bids' : 'asks';

		log.position(`Closing ${currentStr} position`, {
			actualSize: position.size,
		});

		const liquidityCheck = await this.checkLiquidity(side, position.size);
		const validation = this.validateLiquidity(liquidityCheck, true);

		if (!validation.success) {
			log.risk('Close liquidity validation failed, forcing close anyway', {
				error: validation.error,
				slippage: `${validation.slippagePercent.toFixed(3)}%`,
			});
		} else {
			log.position('Close liquidity check passed', {
				side,
				slippage: `${validation.slippagePercent.toFixed(3)}%`,
			});
		}

		const baseAssetAmount = new BN(position.size * BASE_PRECISION_NUM);
		const tx = await this.executeOrder(
			direction,
			baseAssetAmount,
			true,
			'Close'
		);

		// Get close price after order execution
		const btcMarket = driftClient.getPerpMarketAccount(
			btcMarketIndex
		) as PerpMarketAccount;

		const closePrice =
			btcMarket.amm.lastMarkPriceTwap.toNumber() / QUOTE_PRECISION_NUM;

		if (entryPrice > 0) {
			const realizedPnl = this.calculatePositionPnl(
				position.signal,
				position.size,
				entryPrice,
				closePrice
			);
			this.strategyRealizedPnl += realizedPnl;

			log.position('Position closed with realized PnL', {
				entryPrice: entryPrice.toFixed(2),
				closePrice: closePrice.toFixed(2),
				realizedPnl: realizedPnl.toFixed(6),
				totalStrategyPnl: this.strategyRealizedPnl.toFixed(6),
			});
		}

		log.order('CLOSE', 'BTC-PERP', `${currentStr} position closed`, {
			txSignature: tx,
			quantity: position.size,
			marketIndex: btcMarketIndex,
			slippage: `${validation.slippagePercent.toFixed(3)}%`,
		});

		this.currentPosition = 0;
	}

	private async capturePerformanceSnapshot(): Promise<PerformanceSnapshot | null> {
		try {
			const user = driftClient.getUser();

			// Track funding payments
			const currentFundingPnl =
				user.getUnrealizedFundingPNL().toNumber() / QUOTE_PRECISION_NUM;
			const fundingDelta = currentFundingPnl - this.lastFundingPnl;
			this.strategyTotalFunding += fundingDelta;
			this.lastFundingPnl = currentFundingPnl;

			const accountEquity =
				user.getTotalCollateral().toNumber() / QUOTE_PRECISION_NUM;
			const accountUnrealizedPnl =
				user.getUnrealizedPNL().toNumber() / QUOTE_PRECISION_NUM;

			const position = this.getCurrentBTCPosition();
			const btcMarket = driftClient.getPerpMarketAccount(
				btcMarketIndex
			) as PerpMarketAccount;
			const markPrice =
				btcMarket.amm.lastMarkPriceTwap.toNumber() / QUOTE_PRECISION_NUM;

			const trueStrategyPnl =
				this.strategyRealizedPnl + this.strategyTotalFunding;
			const trueStrategyEquity = this.strategyStartingEquity + trueStrategyPnl;

			return {
				timestamp: Date.now(),
				cycle: this.cycleCount,
				accountEquity,
				accountUnrealizedPnl,
				strategyEquity: trueStrategyEquity,
				strategyRealizedPnl: trueStrategyPnl,
				strategyTotalFunding: this.strategyTotalFunding,
				strategyFundingDelta: fundingDelta,
				position: {
					size: position.size,
					entryPrice: position.signal !== 0 ? this.getPositionEntryPrice() : 0,
					markPrice,
				},
			};
		} catch (error) {
			log.error('PERFORMANCE', 'Failed to capture snapshot', error as Error);
			return null;
		}
	}
}
