import type { CandleData, Signal } from './types';

async function fetchBTCCandles(): Promise<CandleData[]> {
	const response = await fetch(
		'https://data.api.drift.trade/market/BTC-PERP/candles/1?limit=500'
	);
	const data = (await response.json()) as any;
	return data.records;
}

function createVolumeFilter(volumes: number[], period: number = 20): boolean[] {
	const filter: boolean[] = new Array(volumes.length);
	let rollingSum = 0;

	for (let j = 0; j < Math.min(period, volumes.length); j++) {
		rollingSum += volumes[j]!;
	}

	for (let i = 0; i < volumes.length; i++) {
		if (i < period - 1) {
			filter[i] = false;
		} else {
			if (i > period - 1) {
				rollingSum = rollingSum - volumes[i - period]! + volumes[i]!;
			}

			const avgVolume = rollingSum / period;
			filter[i] = volumes[i]! > avgVolume * 2;
		}
	}

	return filter;
}

function calculateEMA(prices: number[], period: number): number[] {
	if (prices.length === 0) return [];

	const alpha = 2 / (period + 1);
	const oneMinusAlpha = 1 - alpha;
	const result: number[] = new Array(prices.length);
	result[0] = prices[0]!;

	for (let i = 1; i < prices.length; i++) {
		result[i] = alpha * prices[i]! + oneMinusAlpha * result[i - 1]!;
	}

	return result;
}

function generateMomentumSignal(
	fastEMA: number[],
	slowEMA: number[],
	volumeFilter: boolean[]
): number[] {
	const signals: number[] = new Array(fastEMA.length);

	for (let i = 0; i < fastEMA.length; i++) {
		const momentum = fastEMA[i]! - slowEMA[i]!;
		// let signal = momentum > 0 ? 1 : momentum < 0 ? -1 : 0;
		let signal = momentum > 0 ? -1 : momentum < 0 ? 1 : 0;

		if (signal !== 0 && !volumeFilter[i]!) {
			signal = 0;
		}

		signals[i] = signal;
	}

	return signals;
}

async function generateSignal(): Promise<Signal> {
	const candles = await fetchBTCCandles();
	const minRequired = 119;
	if (candles.length < minRequired) return 0;

	const length = candles.length;
	const prices: number[] = new Array(length);
	const volumes: number[] = new Array(length);

	for (let i = 0; i < length; i++) {
		const candle = candles[i]!;
		prices[i] = candle.oracleClose;
		volumes[i] = candle.baseVolume;
	}

	const volumeFilter = createVolumeFilter(volumes);
	const individualSignals: number[] = new Array(5);

	// const strategies = [
	// 	{ fast: 15, slow: 108 },
	// 	{ fast: 18, slow: 99 },
	// 	{ fast: 15, slow: 116 },
	// 	{ fast: 19, slow: 99 },
	// 	{ fast: 16, slow: 102 },
	// ];

	const strategies = [
		{ fast: 6, slow: 19 },
		{ fast: 6, slow: 22 },
		{ fast: 8, slow: 21 },
		{ fast: 4, slow: 26 },
		{ fast: 4, slow: 23 },
	];

	const lagIdx = length - 3;

	for (let s = 0; s < 5; s++) {
		const config = strategies[s]!;
		const fastEMA = calculateEMA(prices, config.fast);
		const slowEMA = calculateEMA(prices, config.slow);

		const momentumSignals = generateMomentumSignal(
			fastEMA,
			slowEMA,
			volumeFilter
		);

		individualSignals[s] = lagIdx >= 0 ? momentumSignals[lagIdx]! : 0;
	}

	individualSignals.sort((a, b) => a - b);
	const median = individualSignals[2]!;
	const ensembleSignal: Signal = median > 0 ? 1 : median < 0 ? -1 : 0;

	return ensembleSignal;
}

export class SignalCache {
	private currentSignal: Signal = 0;
	private lastUpdate: number = 0;
	private isUpdating: boolean = false;

	public async getSignal(): Promise<{ signal: Signal; age: number }> {
		// const now = Date.now();
		// const signalAge = now - this.lastUpdate;

		// if (signalAge > 55000 && !this.isUpdating) {
		//   this.isUpdating = true;
		//   try {
		//     this.currentSignal = await generateSignal();
		//     this.lastUpdate = now;
		//   } finally {
		//     this.isUpdating = false;
		//   }
		// }

		this.currentSignal = await generateSignal();
		this.lastUpdate = Date.now();

		return {
			signal: this.currentSignal,
			age: 0, // Always fresh
		};

		// return {
		//   signal: this.currentSignal,
		//   age: signalAge,
		// };
	}
}
