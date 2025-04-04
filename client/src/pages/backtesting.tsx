import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchBacktestResults } from '@/lib/ai-model';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { fetchAvailableSymbols } from '@/lib/binance';
import { TIMEFRAMES, STRATEGIES } from '@/lib/constants';
import { BacktestResults } from '@/components/backtest-results';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

export default function Backtesting() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [backTestParams, setBackTestParams] = useState({
    symbol: 'BTC/USDT',
    timeframe: '1h',
    startDate: '2023-01-01',
    endDate: '2023-04-30',
    initialCapital: 10000,
    monteCarloSimulations: 100,
    strategies: STRATEGIES.map(s => ({ id: s.id, isActive: s.id === 'smc' || s.id === 'trend' }))
  });
  const [isRunning, setIsRunning] = useState(false);

  // Fetch available symbols
  const { data: availableSymbols } = useQuery({
    queryKey: [`${API_ENDPOINTS.ACCOUNT}/symbols`],
    queryFn: fetchAvailableSymbols,
  });

  // Fetch backtest results
  const { data: backtestResults = [], isLoading: isLoadingResults } = useQuery({
    queryKey: [`${API_ENDPOINTS.BACKTEST}/results`],
    queryFn: async () => {
      const response = await apiRequest('GET', `${API_ENDPOINTS.BACKTEST}/results`);
      return response.json();
    },
  });

  // Run backtest mutation
  const runBacktestMutation = useMutation({
    mutationFn: async (params: any) => {
      setIsRunning(true);
      const response = await apiRequest('POST', `${API_ENDPOINTS.BACKTEST}/run`, params);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BACKTEST}/results`] });
      toast({
        title: "Backtest completed",
        description: "Backtest has been completed successfully",
      });
      setIsRunning(false);
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to run backtest: ${error}`,
        variant: "destructive",
      });
      setIsRunning(false);
    },
  });

  // Delete backtest result mutation
  const deleteBacktestMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await apiRequest('DELETE', `${API_ENDPOINTS.BACKTEST}/results/${id}`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BACKTEST}/results`] });
      toast({
        title: "Backtest deleted",
        description: "Backtest result has been deleted successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to delete backtest result: ${error}`,
        variant: "destructive",
      });
    },
  });

  const handleRunBacktest = () => {
    // Filter only active strategies
    const activeStrategies = backTestParams.strategies
      .filter(s => s.isActive)
      .map(s => ({ id: s.id }));
    
    if (activeStrategies.length === 0) {
      toast({
        title: "No strategies selected",
        description: "Please select at least one strategy to run the backtest",
        variant: "destructive",
      });
      return;
    }

    runBacktestMutation.mutate({
      ...backTestParams,
      strategies: activeStrategies
    });
  };

  const handleDeleteBacktest = (id: string) => {
    if (confirm("Are you sure you want to delete this backtest result?")) {
      deleteBacktestMutation.mutate(id);
    }
  };

  const updateParam = (key: string, value: any) => {
    setBackTestParams({ ...backTestParams, [key]: value });
  };

  const toggleStrategy = (strategyId: string, isActive: boolean) => {
    const updatedStrategies = backTestParams.strategies.map(s => 
      s.id === strategyId ? { ...s, isActive } : s
    );
    setBackTestParams({ ...backTestParams, strategies: updatedStrategies });
  };

  return (
    <div className="p-4 lg:p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Backtesting</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card className="border-border bg-surface sticky top-4">
            <CardHeader>
              <CardTitle>Backtest Configuration</CardTitle>
              <CardDescription>Configure your backtest parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="symbol" className="mb-2 block">Trading Pair</Label>
                <Select 
                  value={backTestParams.symbol} 
                  onValueChange={(value) => updateParam('symbol', value)}
                >
                  <SelectTrigger id="symbol" className="bg-surfaceLight">
                    <SelectValue placeholder="Select trading pair" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableSymbols?.map(symbol => (
                      <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="timeframe" className="mb-2 block">Timeframe</Label>
                <Select 
                  value={backTestParams.timeframe} 
                  onValueChange={(value) => updateParam('timeframe', value)}
                >
                  <SelectTrigger id="timeframe" className="bg-surfaceLight">
                    <SelectValue placeholder="Select timeframe" />
                  </SelectTrigger>
                  <SelectContent>
                    {TIMEFRAMES.map(tf => (
                      <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="start-date" className="mb-2 block">Start Date</Label>
                <Input 
                  id="start-date" 
                  type="date" 
                  value={backTestParams.startDate}
                  onChange={(e) => updateParam('startDate', e.target.value)}
                  className="bg-surfaceLight"
                />
              </div>

              <div>
                <Label htmlFor="end-date" className="mb-2 block">End Date</Label>
                <Input 
                  id="end-date" 
                  type="date" 
                  value={backTestParams.endDate}
                  onChange={(e) => updateParam('endDate', e.target.value)}
                  className="bg-surfaceLight"
                />
              </div>

              <div>
                <Label htmlFor="initial-capital" className="mb-2 block">Initial Capital (USDT)</Label>
                <Input 
                  id="initial-capital" 
                  type="number" 
                  value={backTestParams.initialCapital}
                  onChange={(e) => updateParam('initialCapital', parseFloat(e.target.value))}
                  className="bg-surfaceLight"
                />
              </div>

              <div>
                <Label className="mb-2 block">Strategies</Label>
                <div className="space-y-2 bg-surfaceLight p-3 rounded-md">
                  {STRATEGIES.map((strategy) => (
                    <div key={strategy.id} className="flex items-center justify-between">
                      <Label htmlFor={`strategy-${strategy.id}`} className="cursor-pointer">
                        {strategy.name}
                      </Label>
                      <Switch 
                        id={`strategy-${strategy.id}`}
                        checked={backTestParams.strategies.find(s => s.id === strategy.id)?.isActive || false}
                        onCheckedChange={(checked) => toggleStrategy(strategy.id, checked)}
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label htmlFor="monte-carlo">Monte Carlo Simulation</Label>
                  <Switch 
                    id="monte-carlo"
                    checked={backTestParams.monteCarloSimulations > 0}
                    onCheckedChange={(checked) => updateParam('monteCarloSimulations', checked ? 100 : 0)}
                  />
                </div>
                {backTestParams.monteCarloSimulations > 0 && (
                  <div>
                    <Label htmlFor="monte-carlo-sims" className="mb-2 block">Number of Simulations</Label>
                    <Input 
                      id="monte-carlo-sims" 
                      type="number" 
                      value={backTestParams.monteCarloSimulations}
                      onChange={(e) => updateParam('monteCarloSimulations', parseInt(e.target.value))}
                      className="bg-surfaceLight"
                    />
                  </div>
                )}
              </div>

              <Button 
                className="w-full bg-primary hover:bg-primary/90" 
                onClick={handleRunBacktest}
                disabled={isRunning}
              >
                {isRunning ? (
                  <span className="flex items-center">
                    <span className="loading-indicator mr-2"></span>
                    Running Backtest...
                  </span>
                ) : 'Run Backtest'}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <BacktestResults 
            results={backtestResults} 
            onDelete={handleDeleteBacktest} 
          />
        </div>
      </div>
    </div>
  );
}
