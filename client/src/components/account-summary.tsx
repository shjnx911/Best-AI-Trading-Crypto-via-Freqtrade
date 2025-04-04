import { useState, useEffect } from 'react';
import { fetchAccountInfo } from '@/lib/binance';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { AccountInfo, DailyProfit, TradingPerformance } from '@shared/types';
import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';

export function AccountSummary() {
  const { data: accountInfo, isLoading: isLoadingAccount } = useQuery<AccountInfo>({
    queryKey: [API_ENDPOINTS.ACCOUNT],
  });

  const { data: dailyProfit, isLoading: isLoadingProfit } = useQuery<DailyProfit>({
    queryKey: [`${API_ENDPOINTS.ACCOUNT}/daily-profit`],
  });

  const { data: tradingPerformance, isLoading: isLoadingPerformance } = useQuery<TradingPerformance>({
    queryKey: [`${API_ENDPOINTS.ACCOUNT}/performance`],
  });

  const isLoading = isLoadingAccount || isLoadingProfit || isLoadingPerformance;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      {/* Balance Card */}
      <div className="bg-surface rounded-lg p-4 border border-border">
        <div className="flex justify-between mb-3">
          <h3 className="text-sm font-medium text-textSecondary">Total Account Value</h3>
          <i className="ri-wallet-3-line text-primary"></i>
        </div>
        {isLoadingAccount ? (
          <div className="animate-pulse h-6 bg-surfaceLight rounded w-32 mb-4"></div>
        ) : (
          <div className="flex items-baseline">
            <span className="text-2xl font-semibold font-mono">
              {formatCurrency(accountInfo?.totalBalance || 0)}
            </span>
            <span className={`ml-2 text-sm ${accountInfo && accountInfo.unrealizedPnlPercent > 0 ? 'text-primary' : 'text-secondary'}`}>
              {accountInfo && accountInfo.unrealizedPnlPercent > 0 ? '+' : ''}
              {formatPercentage(accountInfo?.unrealizedPnlPercent || 0)}
            </span>
          </div>
        )}
        
        <div className="mt-4">
          {isLoadingAccount ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-32"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-20"></div>
              </div>
              <div className="flex justify-between items-center">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-24"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-20"></div>
              </div>
              <div className="flex justify-between items-center">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-28"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-20"></div>
              </div>
            </div>
          ) : (
            <>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-textSecondary">Available Balance</span>
                <span className="font-mono">{formatCurrency(accountInfo?.availableBalance || 0)}</span>
              </div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-textSecondary">In Position</span>
                <span className="font-mono">{formatCurrency(accountInfo ? (accountInfo.totalBalance - accountInfo.availableBalance) : 0)}</span>
              </div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-textSecondary">Unrealized PnL</span>
                <span className={`font-mono ${accountInfo && accountInfo.unrealizedPnl > 0 ? 'text-primary' : 'text-secondary'}`}>
                  {accountInfo && accountInfo.unrealizedPnl > 0 ? '+' : ''}
                  {formatCurrency(accountInfo?.unrealizedPnl || 0)}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Profit Card */}
      <div className="bg-surface rounded-lg p-4 border border-border">
        <div className="flex justify-between mb-3">
          <h3 className="text-sm font-medium text-textSecondary">Daily Profit</h3>
          <i className="ri-line-chart-line text-primary"></i>
        </div>
        {isLoadingProfit ? (
          <div className="animate-pulse h-6 bg-surfaceLight rounded w-32 mb-4"></div>
        ) : (
          <div className="flex items-baseline">
            <span className={`text-2xl font-semibold font-mono ${dailyProfit && dailyProfit.amount > 0 ? 'text-primary' : 'text-secondary'}`}>
              {dailyProfit && dailyProfit.amount > 0 ? '+' : ''}
              {formatCurrency(dailyProfit?.amount || 0)}
            </span>
            <span className={`ml-2 text-sm ${dailyProfit && dailyProfit.percentage > 0 ? 'text-primary' : 'text-secondary'}`}>
              {dailyProfit && dailyProfit.percentage > 0 ? '+' : ''}
              {formatPercentage(dailyProfit?.percentage || 0)}
            </span>
          </div>
        )}
        <div className="flex items-center justify-between mt-4">
          {isLoadingProfit ? (
            <div className="w-full space-y-2">
              <div className="animate-pulse h-2 bg-surfaceLight rounded w-full"></div>
              <div className="flex justify-between">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-24"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-20"></div>
              </div>
            </div>
          ) : (
            <>
              <div className="w-2/3">
                <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-primary h-full rounded-full" 
                    style={{ width: `${dailyProfit?.progress || 0}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span className="text-textSecondary">Daily Goal</span>
                  <span>{dailyProfit?.progress || 0}% ({formatCurrency(dailyProfit?.amount || 0)}/{formatCurrency(dailyProfit?.goal || 0)})</span>
                </div>
              </div>
              <div className="text-right">
                <span className="block text-xl font-medium font-mono">
                  {dailyProfit?.winningDays || 0}/{dailyProfit?.totalDays || 0}
                </span>
                <span className="text-xs text-textSecondary">Winning Days</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Win Rate Card */}
      <div className="bg-surface rounded-lg p-4 border border-border">
        <div className="flex justify-between mb-3">
          <h3 className="text-sm font-medium text-textSecondary">Trading Performance</h3>
          <i className="ri-percent-line text-primary"></i>
        </div>
        {isLoadingPerformance ? (
          <div className="animate-pulse h-6 bg-surfaceLight rounded w-32 mb-4"></div>
        ) : (
          <div className="flex items-baseline">
            <span className="text-2xl font-semibold">Win Rate</span>
            <span className="ml-2 text-xl font-mono text-primary">{formatPercentage(tradingPerformance?.winRate || 0)}</span>
          </div>
        )}
        <div className="mt-4 flex items-center justify-between">
          {isLoadingPerformance ? (
            <div className="w-full space-y-2">
              <div className="flex justify-between">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-32"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-12"></div>
              </div>
              <div className="flex justify-between">
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-28"></div>
                <div className="animate-pulse h-4 bg-surfaceLight rounded w-12"></div>
              </div>
            </div>
          ) : (
            <>
              <div>
                <div className="flex items-center mb-1">
                  <div className="w-3 h-3 bg-primary rounded-sm mr-2"></div>
                  <span className="text-sm">Winning Trades</span>
                  <span className="text-sm ml-2 font-mono">{tradingPerformance?.winningTrades || 0}</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-secondary rounded-sm mr-2"></div>
                  <span className="text-sm">Losing Trades</span>
                  <span className="text-sm ml-2 font-mono">{tradingPerformance?.losingTrades || 0}</span>
                </div>
              </div>
              <div className="flex flex-col items-end">
                <span className="font-mono text-lg">{tradingPerformance?.riskRewardRatio.toFixed(2) || '0.00'}:1</span>
                <span className="text-xs text-textSecondary">Risk-Reward Ratio</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
