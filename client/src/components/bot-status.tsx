import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchBotStatus, toggleBot } from '@/lib/binance';
import { fetchAIModelStats, fetchAIInsights } from '@/lib/ai-model';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { AIModelStats, AIInsight } from '@shared/types';
import { useToast } from '@/hooks/use-toast';

interface StrategyToggleProps {
  name: string;
  isActive: boolean;
  winRate: number;
  onToggle: (isActive: boolean) => void;
}

function StrategyToggle({ name, isActive, winRate, onToggle }: StrategyToggleProps) {
  return (
    <div className="flex justify-between items-center py-1.5 px-3 bg-surfaceLight rounded-md">
      <span>{name}</span>
      <div className="flex items-center">
        <span className={`text-xs mr-2 ${isActive ? 'text-primary' : 'text-textSecondary'}`}>
          {isActive ? `${winRate}% Win` : 'Inactive'}
        </span>
        <Switch
          checked={isActive}
          onCheckedChange={onToggle}
          className="scale-75 origin-right"
        />
      </div>
    </div>
  );
}

export function BotStatus() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch bot status
  const { data: botStatus, isLoading: isLoadingStatus } = useQuery({
    queryKey: [`${API_ENDPOINTS.BOT}/status`],
    queryFn: fetchBotStatus,
  });

  // Fetch AI model stats
  const { data: aiStats, isLoading: isLoadingAI } = useQuery<AIModelStats>({
    queryKey: [`${API_ENDPOINTS.AI}/stats`],
    queryFn: fetchAIModelStats,
  });

  // Fetch AI insights
  const { data: aiInsights, isLoading: isLoadingInsights } = useQuery<AIInsight[]>({
    queryKey: [`${API_ENDPOINTS.AI}/insights`],
    queryFn: fetchAIInsights,
  });

  // Toggle bot mutation
  const toggleBotMutation = useMutation({
    mutationFn: toggleBot,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BOT}/status`] });
      toast({
        title: botStatus?.isRunning ? "Bot Stopped" : "Bot Started",
        description: botStatus?.isRunning 
          ? "The trading bot has been stopped successfully" 
          : "The trading bot has been started successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to ${botStatus?.isRunning ? 'stop' : 'start'} the bot: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Toggle strategy mutation
  const toggleStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => {
      // This would typically update the strategy in the bot configuration
      return Promise.resolve();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BOT}/status`] });
    },
  });

  // Handle bot toggle
  const handleToggleBot = () => {
    toggleBotMutation.mutate(!botStatus?.isRunning);
  };

  // Handle restart bot
  const handleRestartBot = () => {
    if (confirm("Are you sure you want to restart the bot?")) {
      toggleBotMutation.mutate(false);
      setTimeout(() => {
        toggleBotMutation.mutate(true);
      }, 1000);
    }
  };

  const isLoading = isLoadingStatus || isLoadingAI || isLoadingInsights;

  if (isLoading) {
    return (
      <div>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Bot Status</h2>
          <div className="animate-pulse h-9 bg-surfaceLight rounded w-28"></div>
        </div>
        
        <div className="bg-surface rounded-lg border border-border p-4 h-[calc(100%-2rem)]">
          <div className="animate-pulse space-y-4">
            <div className="flex justify-between items-center">
              <div className="h-6 bg-surfaceLight rounded w-32"></div>
              <div className="h-6 bg-surfaceLight rounded w-12"></div>
            </div>
            
            <div>
              <div className="h-5 bg-surfaceLight rounded w-40 mb-3"></div>
              <div className="space-y-2">
                <div className="h-10 bg-surfaceLight rounded"></div>
                <div className="h-10 bg-surfaceLight rounded"></div>
                <div className="h-10 bg-surfaceLight rounded"></div>
                <div className="h-10 bg-surfaceLight rounded"></div>
              </div>
            </div>
            
            <div className="h-36 bg-surfaceLight rounded"></div>
            <div className="h-36 bg-surfaceLight rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Bot Status</h2>
        <div>
          <Button onClick={handleRestartBot} className="bg-primary text-white hover:bg-primary/80">
            <i className="ri-restart-line mr-1"></i>
            Restart Bot
          </Button>
        </div>
      </div>
      
      <div className="bg-surface rounded-lg border border-border p-4 h-[calc(100%-2rem)]">
        <div className="flex justify-between mb-4 items-center">
          <div className="flex items-center">
            <span className={`w-2 h-2 rounded-full mr-2 ${botStatus?.isRunning ? 'bg-primary' : 'bg-secondary'}`}></span>
            <span className="font-medium">
              Bot is {botStatus?.isRunning ? 'running' : 'stopped'}
            </span>
          </div>
          <Switch 
            checked={!!botStatus?.isRunning} 
            onCheckedChange={handleToggleBot}
            disabled={toggleBotMutation.isPending}
          />
        </div>
        
        <div className="space-y-4">
          {/* Active Strategies */}
          <div>
            <h4 className="text-sm text-textSecondary mb-2">Active Strategies</h4>
            <div className="space-y-2">
              <StrategyToggle
                name="Smart Money Concept"
                isActive={botStatus?.activeStrategies.includes('smc') || false}
                winRate={68}
                onToggle={() => toggleStrategyMutation.mutate('smc')}
              />
              <StrategyToggle
                name="Trend Following"
                isActive={botStatus?.activeStrategies.includes('trend') || false}
                winRate={72}
                onToggle={() => toggleStrategyMutation.mutate('trend')}
              />
              <StrategyToggle
                name="VWAP Mean Reversion"
                isActive={botStatus?.activeStrategies.includes('vwap') || false}
                winRate={64}
                onToggle={() => toggleStrategyMutation.mutate('vwap')}
              />
              <StrategyToggle
                name="Breakout Trading"
                isActive={botStatus?.activeStrategies.includes('breakout') || false}
                winRate={58}
                onToggle={() => toggleStrategyMutation.mutate('breakout')}
              />
            </div>
          </div>
          
          {/* Bot Parameters */}
          <div>
            <h4 className="text-sm text-textSecondary mb-2">Current Parameters</h4>
            <div className="space-y-2 text-sm">
              {botStatus?.currentParameters && (
                <>
                  <div className="flex justify-between">
                    <span>Position Size:</span>
                    <span>
                      {botStatus.currentParameters.minPositionSize}-{botStatus.currentParameters.maxPositionSize}% of Capital
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>DCA Levels:</span>
                    <span>{botStatus.currentParameters.dcaLevels}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Take Profit:</span>
                    <span>{botStatus.currentParameters.profitTarget}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Daily Profit Target:</span>
                    <span>{botStatus.currentParameters.dailyProfitTarget}% of Capital</span>
                  </div>
                </>
              )}
            </div>
          </div>
          
          {/* AI Learning Status */}
          <div>
            <h4 className="text-sm text-textSecondary mb-2">AI Learning Status</h4>
            {aiStats && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Model Version:</span>
                  <span>{aiStats.version}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last Training:</span>
                  <span>{aiStats.lastTraining}</span>
                </div>
                <div className="flex justify-between">
                  <span>Training Data:</span>
                  <span>{aiStats.trainingData.toLocaleString()} Trades</span>
                </div>
                <div className="flex justify-between">
                  <span>Accuracy Improvement:</span>
                  <span className="text-primary">+{aiStats.accuracyImprovement}%</span>
                </div>
              </div>
            )}
          </div>
          
          {/* AI Insights */}
          <div className="bg-surfaceLight rounded-md p-3 text-sm">
            <h4 className="text-textSecondary mb-2">Recent AI Insights</h4>
            {aiInsights && aiInsights.length > 0 ? (
              <ul className="space-y-2">
                {aiInsights.map((insight, index) => (
                  <li key={index} className="flex items-center">
                    <div className={`w-1.5 h-1.5 rounded-full mr-2 ${
                      insight.type === 'success' ? 'bg-primary' : 
                      insight.type === 'warning' ? 'bg-warning' : 
                      'bg-secondary'
                    }`}></div>
                    <span>{insight.message}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-textSecondary">No insights available yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
