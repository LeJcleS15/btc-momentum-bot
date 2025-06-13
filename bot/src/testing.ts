import { SignalCache } from './signal';
import type { Signal } from './types';

interface SignalTest {
	timestamp: number;
	signal: Signal;
	priceAtSignal: number;
	price1min: number;
	price5min: number;
	price10min: number;
	move1min: number;
	move5min: number;
	move10min: number;
	correct1min: boolean;
	correct5min: boolean;
	correct10min: boolean;
	signalStrength: number;
}

class SignalValidator {
	private tests: SignalTest[] = [];
	private signalCache = new SignalCache();
	private isRunning = false;

	async startValidation(): Promise<void> {
		console.log('Starting signal validation...');
		this.isRunning = true;

		while (this.isRunning) {
			try {
				await this.collectSignalTest();
				await this.sleep(60000); // test every minute
			} catch (error) {
				console.error('Signal test error:', error);
				await this.sleep(5000);
			}
		}
	}

	private async collectSignalTest(): Promise<void> {
		const startTime = Date.now();

		// get current signal and price
		const signalData = await this.signalCache.getSignal();
		const currentPrice = await this.getCurrentBTCPrice();

		if (!currentPrice) return;

		console.log(
			`Signal: ${this.signalToString(signalData.signal)} at price ${currentPrice}`
		);

		// wait and collect future prices
		const price1min = await this.waitAndGetPrice(60 * 1000);
		const price5min = await this.waitAndGetPrice(4 * 60 * 1000); // additional 4 min
		const price10min = await this.waitAndGetPrice(5 * 60 * 1000); // additional 5 min

		if (!price1min || !price5min || !price10min) return;

		// calculate moves and accuracy
		const move1min = price1min - currentPrice;
		const move5min = price5min - currentPrice;
		const move10min = price10min - currentPrice;

		const correct1min = this.isCorrectPrediction(signalData.signal, move1min);
		const correct5min = this.isCorrectPrediction(signalData.signal, move5min);
		const correct10min = this.isCorrectPrediction(signalData.signal, move10min);

		const test: SignalTest = {
			timestamp: startTime,
			signal: signalData.signal,
			priceAtSignal: currentPrice,
			price1min,
			price5min,
			price10min,
			move1min,
			move5min,
			move10min,
			correct1min,
			correct5min,
			correct10min,
			signalStrength: Math.abs(signalData.signal),
		};

		this.tests.push(test);
		this.logTest(test);
		this.printStats();

		// save to file periodically
		if (this.tests.length % 10 === 0) {
			await this.saveResults();
		}
	}

	private async getCurrentBTCPrice(): Promise<number | null> {
		try {
			const response = await fetch(
				'https://data.api.drift.trade/market/BTC-PERP/candles/1?limit=1'
			);
			const data = (await response.json()) as any;
			return data.records[0]?.oracleClose || null;
		} catch {
			return null;
		}
	}

	private async waitAndGetPrice(ms: number): Promise<number | null> {
		await this.sleep(ms);
		return this.getCurrentBTCPrice();
	}

	private isCorrectPrediction(signal: Signal, actualMove: number): boolean {
		if (signal === 0) return true; // neutral signals always "correct"
		return (signal > 0 && actualMove > 0) || (signal < 0 && actualMove < 0);
	}

	private signalToString(signal: Signal): string {
		return signal > 0 ? 'LONG' : signal < 0 ? 'SHORT' : 'FLAT';
	}

	private logTest(test: SignalTest): void {
		console.log(`
Test Result:
  Signal: ${this.signalToString(test.signal)}
  Price at signal: ${test.priceAtSignal.toFixed(2)}
  1min: ${test.price1min.toFixed(2)} (${test.move1min.toFixed(2)}) ${test.correct1min ? '✓' : '✗'}
  5min: ${test.price5min.toFixed(2)} (${test.move5min.toFixed(2)}) ${test.correct5min ? '✓' : '✗'}
  10min: ${test.price10min.toFixed(2)} (${test.move10min.toFixed(2)}) ${test.correct10min ? '✓' : '✗'}
    `);
	}

	private printStats(): void {
		if (this.tests.length === 0) return;

		const totalTests = this.tests.length;
		const nonFlatTests = this.tests.filter((t) => t.signal !== 0);

		if (nonFlatTests.length === 0) {
			console.log(`Stats: ${totalTests} tests, all FLAT signals`);
			return;
		}

		const accuracy1min =
			nonFlatTests.filter((t) => t.correct1min).length / nonFlatTests.length;
		const accuracy5min =
			nonFlatTests.filter((t) => t.correct5min).length / nonFlatTests.length;
		const accuracy10min =
			nonFlatTests.filter((t) => t.correct10min).length / nonFlatTests.length;

		const avgMove1min =
			nonFlatTests.reduce((sum, t) => sum + Math.abs(t.move1min), 0) /
			nonFlatTests.length;
		const avgMove5min =
			nonFlatTests.reduce((sum, t) => sum + Math.abs(t.move5min), 0) /
			nonFlatTests.length;
		const avgMove10min =
			nonFlatTests.reduce((sum, t) => sum + Math.abs(t.move10min), 0) /
			nonFlatTests.length;

		// expected value calculation
		const ev1min = this.calculateExpectedValue(
			nonFlatTests,
			'move1min',
			'correct1min'
		);
		const ev5min = this.calculateExpectedValue(
			nonFlatTests,
			'move5min',
			'correct5min'
		);
		const ev10min = this.calculateExpectedValue(
			nonFlatTests,
			'move10min',
			'correct10min'
		);

		console.log(`
=== SIGNAL VALIDATION STATS (${totalTests} total tests, ${nonFlatTests.length} directional) ===
Accuracy:  1min: ${(accuracy1min * 100).toFixed(1)}%  |  5min: ${(accuracy5min * 100).toFixed(1)}%  |  10min: ${(accuracy10min * 100).toFixed(1)}%
Avg Move:  1min: ${avgMove1min.toFixed(2)}  |  5min: ${avgMove5min.toFixed(2)}  |  10min: ${avgMove10min.toFixed(2)}
Expected Value: 1min: ${ev1min.toFixed(2)}  |  5min: ${ev5min.toFixed(2)}  |  10min: ${ev10min.toFixed(2)}

Signal Distribution:
  LONG: ${this.tests.filter((t) => t.signal > 0).length}
  SHORT: ${this.tests.filter((t) => t.signal < 0).length}  
  FLAT: ${this.tests.filter((t) => t.signal === 0).length}
    `);
	}

	private calculateExpectedValue(
		tests: SignalTest[],
		moveKey: keyof SignalTest,
		correctKey: keyof SignalTest
	): number {
		let totalEV = 0;
		for (const test of tests) {
			const move = test[moveKey] as number;
			const correct = test[correctKey] as boolean;
			// if signal was correct, we make the move; if wrong, we lose the move
			const signedMove = test.signal * move;
			totalEV += signedMove;
		}
		return totalEV / tests.length;
	}

	private async saveResults(): Promise<void> {
		try {
			const filename = `signal_validation_${Date.now()}.json`;
			await Bun.write(filename, JSON.stringify(this.tests, null, 2));
			console.log(`Saved ${this.tests.length} tests to ${filename}`);
		} catch (error) {
			console.error('Failed to save results:', error);
		}
	}

	private sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	stop(): void {
		this.isRunning = false;
		console.log('Signal validation stopped');
	}
}

// Run the test
async function main() {
	const validator = new SignalValidator();

	// Handle shutdown
	process.on('SIGINT', () => {
		console.log('\nShutting down...');
		validator.stop();
		process.exit(0);
	});

	await validator.startValidation();
}

if (import.meta.main) {
	main().catch(console.error);
}
