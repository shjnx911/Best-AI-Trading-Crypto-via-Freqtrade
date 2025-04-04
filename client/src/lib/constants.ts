// Server endpoints
export const API_BASE_URL = '/api';
export const API_ENDPOINTS = {
  ACCOUNT: `${API_BASE_URL}/account`,
  POSITIONS: `${API_BASE_URL}/positions`,
  TRADES: `${API_BASE_URL}/trades`,
  BOT: `${API_BASE_URL}/bot`,
  BACKTEST: `${API_BASE_URL}/backtest`,
  SYSTEM: `${API_BASE_URL}/system`,
  DATA: `${API_BASE_URL}/data`,
  AI: `${API_BASE_URL}/ai`,
};

// Timeframes
export const TIMEFRAMES = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '2h', label: '2 Hours' },
  { value: '4h', label: '4 Hours' },
  { value: '6h', label: '6 Hours' },
  { value: '12h', label: '12 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '3d', label: '3 Days' },
  { value: '1w', label: '1 Week' },
];

// Trading strategies
export const STRATEGIES = [
  { 
    id: 'smc',
    name: 'Smart Money Concept',
    description: 'Based on order flow analysis and liquidity zones identification',
    indicators: ['Order Blocks', 'Liquidity Zones', 'Break of Structure', 'Change of Character']
  },
  { 
    id: 'trend',
    name: 'Trend Following',
    description: 'Follow major market trends using technical indicators',
    indicators: ['EMA', 'MACD', 'RSI', 'VWAP']
  },
  { 
    id: 'hft',
    name: 'High-Frequency Trading',
    description: 'Fast executions based on short-term price movements and arbitrage',
    indicators: ['Order Book Analysis', 'Price Action', 'Volume Profile']
  },
  { 
    id: 'vwap',
    name: 'VWAP Mean Reversion',
    description: 'Buy when price drops below VWAP and sell when it rises above average',
    indicators: ['VWAP', 'Bollinger Bands', 'RSI']
  },
  { 
    id: 'breakout',
    name: 'Breakout Trading',
    description: 'Trade when price breaks through important support/resistance levels',
    indicators: ['Support/Resistance', 'Volume Confirmation', 'Price Patterns']
  },
  { 
    id: 'liquidity',
    name: 'Liquidity Grab & Stop Hunt',
    description: 'Identify areas with accumulated stop orders and trade the reversal',
    indicators: ['Stop Order Clusters', 'Market Maker Behavior', 'Price Pattern Recognition']
  }
];

// Technical indicators
export const INDICATORS = [
  { value: 'rsi', label: 'RSI' },
  { value: 'ema', label: 'EMA' },
  { value: 'macd', label: 'MACD' },
  { value: 'fibonacci', label: 'Fibonacci' },
  { value: 'darvas', label: 'Darvas Box' },
  { value: 'candlestick', label: 'Candlestick Patterns' }
];

// Position sizes
export const POSITION_SIZE_RANGE = {
  min: 1,
  max: 20
};

// DCA levels
export const DCA_LEVELS_RANGE = {
  min: 1,
  max: 10
};

// Profit targets
export const PROFIT_TARGET_RANGE = {
  min: 0.5,
  max: 10
};

// Stop loss
export const STOP_LOSS_RANGE = {
  min: 0.5,
  max: 10
};

// Trailing settings
export const TRAILING_RANGE = {
  min: 0.1,
  max: 5
};

// System statuses
export const SYSTEM_STATUSES = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  MAINTENANCE: 'maintenance'
};

// AI model versions
export const AI_MODEL_VERSIONS = [
  { value: 'v1.0.0', label: 'Basic Model' },
  { value: 'v2.0.0', label: 'Advanced Analysis' },
  { value: 'v2.3.4', label: 'Deep Learning Enhanced' }
];

// Chart themes
export const CHART_THEMES = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' }
];

// Default pagination
export const DEFAULT_PAGE_SIZE = 10;
