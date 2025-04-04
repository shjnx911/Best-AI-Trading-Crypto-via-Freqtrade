import { API_ENDPOINTS } from './constants';
import { apiRequest } from './queryClient';
import { AccountInfo, ActivePosition, TradeHistory } from '@shared/types';

export async function fetchAccountInfo(): Promise<AccountInfo> {
  const response = await apiRequest('GET', API_ENDPOINTS.ACCOUNT);
  return response.json();
}

export async function fetchActivePositions(): Promise<ActivePosition[]> {
  const response = await apiRequest('GET', API_ENDPOINTS.POSITIONS);
  return response.json();
}

export async function closePosition(positionId: number): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.POSITIONS}/${positionId}/close`);
}

export async function updatePosition(positionId: number, data: {
  takeProfit?: number;
  stopLoss?: number;
}): Promise<void> {
  await apiRequest('PATCH', `${API_ENDPOINTS.POSITIONS}/${positionId}`, data);
}

export async function fetchTradingHistory(page = 1, pageSize = 10): Promise<{
  trades: TradeHistory[];
  total: number;
}> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.TRADES}?page=${page}&pageSize=${pageSize}`);
  return response.json();
}

export async function openPosition(data: {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  leverage: number;
  takeProfit?: number;
  stopLoss?: number;
  strategy: string;
}): Promise<void> {
  await apiRequest('POST', API_ENDPOINTS.POSITIONS, data);
}

export async function fetchMarketPrice(symbol: string): Promise<number> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.ACCOUNT}/price?symbol=${symbol}`);
  const data = await response.json();
  return data.price;
}

export async function fetchAvailableSymbols(): Promise<string[]> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.ACCOUNT}/symbols`);
  return response.json();
}

export async function updateBotSettings(settings: any): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.BOT}/settings`, settings);
}

export async function toggleBot(isActive: boolean): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.BOT}/toggle`, { isActive });
}

export async function fetchBotSettings(): Promise<any> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.BOT}/settings`);
  return response.json();
}

export async function fetchBotStatus(): Promise<{
  isRunning: boolean;
  activeStrategies: string[];
  currentParameters: any;
}> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.BOT}/status`);
  return response.json();
}

export async function saveApiKeys(apiKey: string, apiSecret: string, isTestnet: boolean): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.SYSTEM}/api-keys`, {
    apiKey,
    apiSecret,
    isTestnet
  });
}
