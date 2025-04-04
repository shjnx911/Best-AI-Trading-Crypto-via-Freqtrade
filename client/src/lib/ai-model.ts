import { API_ENDPOINTS } from './constants';
import { apiRequest } from './queryClient';
import { AIModelStats, AIInsight, MarketAnalysis } from '@shared/types';

export async function fetchAIModelStats(): Promise<AIModelStats> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.AI}/stats`);
  return response.json();
}

export async function fetchAIInsights(): Promise<AIInsight[]> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.AI}/insights`);
  return response.json();
}

export async function fetchMarketAnalysis(symbol: string, timeframe: string): Promise<MarketAnalysis> {
  const response = await apiRequest(
    'GET', 
    `${API_ENDPOINTS.AI}/market-analysis?symbol=${symbol}&timeframe=${timeframe}`
  );
  return response.json();
}

export async function trainAIModel(options: {
  datasets: string[];
  epochs: number;
  batchSize: number;
}): Promise<void> {
  try {
    const response = await apiRequest('POST', `${API_ENDPOINTS.AI}/train`, options);
    if (!response.ok) {
      throw new Error(`Training failed with status ${response.status}`);
    }
    return response.json();
  } catch (error) {
    console.error("Error starting AI training:", error);
    throw error;
  }
}

export async function pauseAITraining(): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.AI}/pause`, {});
}

export async function resumeAITraining(): Promise<void> {
  await apiRequest('POST', `${API_ENDPOINTS.AI}/resume`, {});
}

export async function getTrainingProgress(): Promise<{
  isTraining: boolean;
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  accuracy: number;
  loss: number;
}> {
  const response = await apiRequest('GET', `${API_ENDPOINTS.AI}/training-progress`);
  return response.json();
}

export async function evaluateStrategy(
  strategy: string,
  symbol: string,
  timeframe: string
): Promise<{
  winProbability: number;
  recommendedSize: number;
  optimalLeverage: number;
  signal: string;
}> {
  const response = await apiRequest('POST', `${API_ENDPOINTS.AI}/evaluate-strategy`, {
    strategy,
    symbol,
    timeframe
  });
  return response.json();
}

export async function analyzeTradePerformance(tradeIds: number[]): Promise<{
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
}> {
  const response = await apiRequest('POST', `${API_ENDPOINTS.AI}/analyze-trades`, {
    tradeIds
  });
  return response.json();
}
