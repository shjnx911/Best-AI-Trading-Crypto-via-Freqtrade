// Trading data interfaces

export interface AccountInfo {
  totalBalance: number;
  availableBalance: number;
  positions: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
}

export interface DailyProfit {
  amount: number;
  percentage: number;
  goal: number;
  progress: number;
  winningDays: number;
  totalDays: number;
}

export interface TradingPerformance {
  winRate: number;
  winningTrades: number;
  losingTrades: number;
  riskRewardRatio: number;
}

export interface ActivePosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  markPrice: number;
  takeProfit: number;
  stopLoss: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
}

export interface TradeHistory {
  id: number;
  timestamp: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  pnlPercent: number;
  strategy: string;
  aiScore: number;
}

// Bot settings types

export interface BotSettings {
  isActive: boolean;
  minPositionSize: number;
  maxPositionSize: number;
  maxPairs: number;
  dcaLevels: number;
  profitTarget: number;
  stopLoss: number;
  dailyProfitTarget: number;
  trailingProfitEnabled: boolean;
  trailingProfitPercent: number;
  trailingStopLossEnabled: boolean;
  trailingStopLossPercent: number;
  activeStrategies: StrategyConfig[];
}

export interface StrategyConfig {
  name: string;
  isActive: boolean;
  winRate: number;
  parameters: Record<string, any>;
}

// Market analysis types

export interface MarketAnalysis {
  symbol: string;
  timeframe: string;
  price: number;
  priceAnalysis: {
    maStatus: string;
    volatility: string;
    support: number;
    resistance: number;
  };
  smcAnalysis: {
    orderBlocks: string;
    liquidity: string;
    bos: string;
    choch: string;
  };
  aiInsights: {
    winProbability: number;
    recommendedSize: number;
    optimalLeverage: number;
    signal: string;
  };
  indicators: {
    rsi: number;
    macd: string;
    ema50: number;
    fibLevel: number;
  };
}

// Backtesting types

export interface BacktestParams {
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  strategies: StrategyConfig[];
  monteCarloSimulations: number;
}

export interface BacktestResult {
  id: string;
  name: string;
  date: string;
  symbol: string;
  timeframe: string;
  initialCapital: number;
  finalCapital: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
  monteCarloPaths?: number;
  monteCarloResults?: any;
}

// System settings types

export interface SystemSettings {
  apiKey: string;
  apiSecret: string;
  isTestnet: boolean;
  telegramEnabled: boolean;
  telegramToken: string;
  telegramChatId: string;
  uiTheme: 'dark' | 'light';
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  backupEnabled: boolean;
  backupFrequency: 'daily' | 'weekly' | 'monthly';
}

// AI model types

export interface AIModelStats {
  version: string;
  lastTraining: string;
  trainingData: number;
  accuracyImprovement: number;
}

export interface AIInsight {
  message: string;
  type: 'success' | 'warning' | 'error';
}

// System status

export interface SystemStatus {
  status: 'online' | 'offline' | 'maintenance';
  cpu: number;
  ram: number;
  gpu: number;
  systemInfo: {
    cpu: string;
    ram: string;
    gpu: string;
  };
}
