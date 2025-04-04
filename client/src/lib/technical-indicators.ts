import { API_ENDPOINTS } from './constants';
import { apiRequest } from './queryClient';

export interface IndicatorParameters {
  symbol: string;
  timeframe: string;
  period?: number;
  [key: string]: any;
}

export interface RSIData {
  timestamp: number[];
  values: number[];
  overbought: number;
  oversold: number;
}

export interface EMAData {
  timestamp: number[];
  values: number[];
}

export interface MACDData {
  timestamp: number[];
  macd: number[];
  signal: number[];
  histogram: number[];
}

export interface FibonacciData {
  timestamp: number[];
  levels: {
    level: number;
    value: number;
  }[];
}

export interface DarvasBoxData {
  timestamp: number[];
  boxes: {
    top: number;
    bottom: number;
    startTime: number;
    endTime: number;
  }[];
}

export interface CandleStickPattern {
  timestamp: number;
  pattern: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  strength: number;
}

// RSI indicator
export async function getRSI(params: IndicatorParameters): Promise<RSIData> {
  const { symbol, timeframe, period = 14 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/rsi?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}

// EMA indicator
export async function getEMA(params: IndicatorParameters): Promise<EMAData> {
  const { symbol, timeframe, period = 50 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/ema?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}

// MACD indicator
export async function getMACD(params: IndicatorParameters): Promise<MACDData> {
  const { symbol, timeframe, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/macd?symbol=${symbol}&timeframe=${timeframe}&fastPeriod=${fastPeriod}&slowPeriod=${slowPeriod}&signalPeriod=${signalPeriod}`
  );
  return response.json();
}

// Fibonacci levels
export async function getFibonacciLevels(params: IndicatorParameters): Promise<FibonacciData> {
  const { symbol, timeframe, startTime, endTime } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/fibonacci?symbol=${symbol}&timeframe=${timeframe}&startTime=${startTime}&endTime=${endTime}`
  );
  return response.json();
}

// Darvas Box
export async function getDarvasBox(params: IndicatorParameters): Promise<DarvasBoxData> {
  const { symbol, timeframe, period = 20 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/darvas?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}

// Candlestick patterns
export async function getCandlestickPatterns(params: IndicatorParameters): Promise<CandleStickPattern[]> {
  const { symbol, timeframe, lookback = 50 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/candlestick-patterns?symbol=${symbol}&timeframe=${timeframe}&lookback=${lookback}`
  );
  return response.json();
}

// Support and resistance levels
export async function getSupportResistanceLevels(params: IndicatorParameters): Promise<{
  timestamp: number[];
  support: number[];
  resistance: number[];
}> {
  const { symbol, timeframe, period = 50 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/support-resistance?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}

// Order blocks
export async function getOrderBlocks(params: IndicatorParameters): Promise<{
  timestamp: number[];
  bullishBlocks: {
    startTime: number;
    endTime: number;
    low: number;
    high: number;
  }[];
  bearishBlocks: {
    startTime: number;
    endTime: number;
    low: number;
    high: number;
  }[];
}> {
  const { symbol, timeframe, period = 50 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/order-blocks?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}

// Volume profile
export async function getVolumeProfile(params: IndicatorParameters): Promise<{
  price: number[];
  volume: number[];
  valueArea: {
    high: number;
    low: number;
  };
  pointOfControl: number;
}> {
  const { symbol, timeframe, period = 50 } = params;
  const response = await apiRequest(
    'GET',
    `${API_ENDPOINTS.DATA}/indicators/volume-profile?symbol=${symbol}&timeframe=${timeframe}&period=${period}`
  );
  return response.json();
}
