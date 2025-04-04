import { BacktestResult } from "@shared/types";

interface BacktestParams {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  strategies: any[];
  monteCarloSimulations?: number;
}

interface DataDownloadParams {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
}

interface AvailableDataItem {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  candles: number;
  sizeInMB: number;
}

export class BacktestService {
  private availableData: AvailableDataItem[] = [
    {
      symbol: "BTC/USDT",
      timeframe: "1h",
      startDate: "2022-01-01",
      endDate: "2023-04-30",
      candles: 11568,
      sizeInMB: 2.3
    },
    {
      symbol: "ETH/USDT",
      timeframe: "1h",
      startDate: "2022-01-01",
      endDate: "2023-04-30",
      candles: 11568,
      sizeInMB: 2.3
    },
    {
      symbol: "BTC/USDT",
      timeframe: "4h",
      startDate: "2022-01-01",
      endDate: "2023-04-30",
      candles: 2892,
      sizeInMB: 0.6
    },
    {
      symbol: "ETH/USDT",
      timeframe: "4h",
      startDate: "2022-01-01",
      endDate: "2023-04-30",
      candles: 2892,
      sizeInMB: 0.6
    }
  ];

  async runBacktest(params: BacktestParams): Promise<BacktestResult> {
    const { symbol, timeframe, startDate, endDate, initialCapital, strategies, monteCarloSimulations } = params;
    
    // This would normally run a real backtest on historical data
    // For now, we'll simulate a backtest result
    
    // Check if we have data for the requested symbol and timeframe
    const dataExists = this.availableData.some(
      item => item.symbol === symbol && item.timeframe === timeframe
    );
    
    if (!dataExists) {
      throw new Error(`No data available for ${symbol} on ${timeframe} timeframe. Please download it first.`);
    }
    
    // Simulate backtest result
    const winRate = 55 + Math.floor(Math.random() * 20);
    const totalTrades = 70 + Math.floor(Math.random() * 50);
    const winningTrades = Math.floor(totalTrades * (winRate / 100));
    const losingTrades = totalTrades - winningTrades;
    
    const returnPercentage = 15 + Math.floor(Math.random() * 25);
    const finalCapital = initialCapital * (1 + (returnPercentage / 100));
    
    const profitFactor = 1.5 + Math.random();
    const maxDrawdown = 5 + Math.floor(Math.random() * 10);
    
    // Generate Monte Carlo simulation results if requested
    let monteCarloResults;
    if (monteCarloSimulations && monteCarloSimulations > 0) {
      monteCarloResults = this.generateMonteCarloResults(returnPercentage, monteCarloSimulations);
    }
    
    const result: BacktestResult = {
      id: "", // Will be set when stored in the database
      name: `${symbol} ${timeframe} Backtest`,
      date: new Date().toISOString(),
      symbol,
      timeframe,
      initialCapital,
      finalCapital,
      totalTrades,
      winningTrades,
      losingTrades,
      winRate,
      profitFactor,
      maxDrawdown,
      monteCarloPaths: monteCarloSimulations,
      monteCarloResults
    };
    
    return result;
  }

  async getAvailableData(): Promise<AvailableDataItem[]> {
    return this.availableData;
  }

  async downloadData(params: DataDownloadParams): Promise<void> {
    const { symbol, timeframe, startDate, endDate } = params;
    
    // This would normally download data from Binance API
    // For now, we'll simulate adding a new data item
    
    const startDateTime = new Date(startDate);
    const endDateTime = new Date(endDate);
    const diffTime = Math.abs(endDateTime.getTime() - startDateTime.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    let candles = 0;
    if (timeframe === "1m") candles = diffDays * 24 * 60;
    else if (timeframe === "5m") candles = diffDays * 24 * 12;
    else if (timeframe === "15m") candles = diffDays * 24 * 4;
    else if (timeframe === "1h") candles = diffDays * 24;
    else if (timeframe === "4h") candles = diffDays * 6;
    else if (timeframe === "1d") candles = diffDays;
    
    const sizeInMB = candles * 0.0002; // Approximate size calculation
    
    // Add to available data if it doesn't already exist
    const exists = this.availableData.some(
      item => item.symbol === symbol && 
              item.timeframe === timeframe && 
              item.startDate === startDate &&
              item.endDate === endDate
    );
    
    if (!exists) {
      this.availableData.push({
        symbol,
        timeframe,
        startDate,
        endDate,
        candles,
        sizeInMB
      });
    }
  }

  async deleteData(symbol: string, timeframe: string): Promise<void> {
    // Remove data item
    this.availableData = this.availableData.filter(
      item => !(item.symbol === symbol && item.timeframe === timeframe)
    );
  }

  private generateMonteCarloResults(baseReturn: number, paths: number): any {
    const returns = [];
    
    for (let i = 0; i < paths; i++) {
      // Generate a random return around the base return
      const randomOffset = (Math.random() - 0.5) * (baseReturn * 0.8);
      returns.push(baseReturn + randomOffset);
    }
    
    const sortedReturns = [...returns].sort((a, b) => a - b);
    
    // Calculate statistics
    const avgReturn = returns.reduce((sum, val) => sum + val, 0) / paths;
    const minReturn = sortedReturns[0];
    const maxReturn = sortedReturns[sortedReturns.length - 1];
    
    // Calculate standard deviation
    const sumSquaredDiff = returns.reduce((sum, val) => sum + Math.pow(val - avgReturn, 2), 0);
    const standardDeviation = Math.sqrt(sumSquaredDiff / paths);
    
    // Calculate percentiles
    const percentile10 = sortedReturns[Math.floor(paths * 0.1)];
    const percentile25 = sortedReturns[Math.floor(paths * 0.25)];
    const percentile50 = sortedReturns[Math.floor(paths * 0.5)];
    const percentile75 = sortedReturns[Math.floor(paths * 0.75)];
    const percentile90 = sortedReturns[Math.floor(paths * 0.9)];
    
    return {
      returns: sortedReturns,
      avgReturn,
      minReturn,
      maxReturn,
      standardDeviation,
      percentiles: {
        10: percentile10,
        25: percentile25,
        50: percentile50,
        75: percentile75,
        90: percentile90
      }
    };
  }
}
