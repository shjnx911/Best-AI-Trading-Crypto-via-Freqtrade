import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { TIMEFRAMES } from '@/lib/constants';
import { fetchMarketAnalysis } from '@/lib/ai-model';
import { formatCurrency } from '@/lib/utils';
import { Chart } from '@/components/ui/chart';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { MarketAnalysis as MarketAnalysisType } from '@shared/types';

// Mock chart data for demonstration
const mockChartData = [
  { time: '2023-05-18', open: 27100, high: 27800, low: 26900, close: 27450 },
  { time: '2023-05-19', open: 27450, high: 28200, low: 27300, close: 27950 },
  { time: '2023-05-20', open: 27950, high: 28100, low: 27500, close: 27650 },
  { time: '2023-05-21', open: 27650, high: 27950, low: 27400, close: 27850 },
  { time: '2023-05-22', open: 27850, high: 28500, low: 27800, close: 28350 },
  // More chart data would be added here
];

export function MarketAnalysis() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');

  // Fetch market analysis
  const { data: analysis, isLoading } = useQuery<MarketAnalysisType>({
    queryKey: [`${API_ENDPOINTS.AI}/market-analysis`, selectedSymbol, selectedTimeframe],
    queryFn: () => fetchMarketAnalysis(selectedSymbol, selectedTimeframe),
  });

  if (isLoading) {
    return (
      <div className="lg:col-span-2">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Market Analysis</h2>
          <div className="animate-pulse flex space-x-2">
            <div className="h-9 bg-surfaceLight rounded w-28"></div>
            <div className="h-9 bg-surfaceLight rounded w-28"></div>
          </div>
        </div>
        
        <div className="bg-surface rounded-lg border border-border p-4">
          <div className="animate-pulse">
            <div className="h-[300px] bg-surfaceLight rounded"></div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="h-32 bg-surfaceLight rounded"></div>
              <div className="h-32 bg-surfaceLight rounded"></div>
              <div className="h-32 bg-surfaceLight rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="lg:col-span-2">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Market Analysis</h2>
        <div className="flex items-center text-sm">
          <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
            <SelectTrigger className="w-[120px] bg-surfaceLight mr-2">
              <SelectValue placeholder="Symbol" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
              <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
              <SelectItem value="SOL/USDT">SOL/USDT</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-[120px] bg-surfaceLight">
              <SelectValue placeholder="Timeframe" />
            </SelectTrigger>
            <SelectContent>
              {TIMEFRAMES.map(tf => (
                <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      
      <div className="bg-surface rounded-lg border border-border p-4">
        <div className="chart-container">
          <Chart
            data={mockChartData}
            chartType="candle"
            height={300}
            indicators={[
              { name: `RSI: ${analysis?.indicators.rsi}`, data: [], color: '#8E59FF' },
              { name: `MACD: ${analysis?.indicators.macd}`, data: [], color: '#F0B90B' },
              { name: `EMA(50): ${formatCurrency(analysis?.indicators.ema50 || 0)}`, data: [], color: '#2962FF' },
              { name: `Fib Level: ${analysis?.indicators.fibLevel}`, data: [], color: '#00B7C2' }
            ]}
          />
          
          {analysis && (
            <div className="absolute top-4 right-4 bg-primary/10 text-primary text-xs px-2 py-1 rounded flex items-center">
              <i className="ri-ai-generate mr-1"></i>
              <span>AI Prediction: {analysis.aiInsights.signal}</span>
            </div>
          )}
        </div>
        
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          {/* Price Analysis */}
          <div className="bg-surfaceLight rounded-md p-3">
            <h4 className="text-textSecondary mb-2">Price Analysis</h4>
            {analysis && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>MA Status:</span>
                  <span className={analysis.priceAnalysis.maStatus === 'Bullish' ? 'text-primary' : 'text-secondary'}>
                    {analysis.priceAnalysis.maStatus}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Volatility:</span>
                  <span>{analysis.priceAnalysis.volatility}</span>
                </div>
                <div className="flex justify-between">
                  <span>Support:</span>
                  <span className="font-mono">{formatCurrency(analysis.priceAnalysis.support)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Resistance:</span>
                  <span className="font-mono">{formatCurrency(analysis.priceAnalysis.resistance)}</span>
                </div>
              </div>
            )}
          </div>
          
          {/* SMC Analysis */}
          <div className="bg-surfaceLight rounded-md p-3">
            <h4 className="text-textSecondary mb-2">SMC Analysis</h4>
            {analysis && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Order Blocks:</span>
                  <span className={analysis.smcAnalysis.orderBlocks === 'Bullish' ? 'text-primary' : 'text-secondary'}>
                    {analysis.smcAnalysis.orderBlocks}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Liquidity:</span>
                  <span>{analysis.smcAnalysis.liquidity}</span>
                </div>
                <div className="flex justify-between">
                  <span>BOS:</span>
                  <span>{analysis.smcAnalysis.bos}</span>
                </div>
                <div className="flex justify-between">
                  <span>ChoCh:</span>
                  <span>{analysis.smcAnalysis.choch}</span>
                </div>
              </div>
            )}
          </div>
          
          {/* AI Insights */}
          <div className="bg-surfaceLight rounded-md p-3">
            <h4 className="text-textSecondary mb-2">AI Insights</h4>
            {analysis && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Win Probability:</span>
                  <span className="text-primary">{analysis.aiInsights.winProbability}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Recommended Size:</span>
                  <span>{analysis.aiInsights.recommendedSize}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Optimal Leverage:</span>
                  <span>{analysis.aiInsights.optimalLeverage}x</span>
                </div>
                <div className="flex justify-between">
                  <span>Signal:</span>
                  <span className={`font-mono ${analysis.aiInsights.signal === 'Long' ? 'text-primary' : 'text-secondary'}`}>
                    {analysis.aiInsights.signal}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
