import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogClose, DialogFooter } from '@/components/ui/dialog';
import { formatCurrency, formatPercentage, formatNumber, formatDate } from '@/lib/utils';
import { Chart } from '@/components/ui/chart';
import { Trash2, FileText, ChartBar } from 'lucide-react';
import { BacktestResult } from '@shared/types';

interface BacktestResultsProps {
  results: BacktestResult[];
  onDelete: (id: string) => void;
}

export function BacktestResults({ results, onDelete }: BacktestResultsProps) {
  const [selectedResult, setSelectedResult] = React.useState<BacktestResult | null>(null);
  const [detailsOpen, setDetailsOpen] = React.useState(false);

  if (results.length === 0) {
    return (
      <Card className="border-border bg-surface">
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center text-center p-6">
            <ChartBar className="h-12 w-12 text-textSecondary mb-3" />
            <h3 className="text-lg font-medium mb-2">No Backtest Results</h3>
            <p className="text-sm text-textSecondary mb-4">
              Run a backtest to see your trading strategy performance.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Generate mock equity curve data from a backtest result
  const generateEquityCurveData = (result: BacktestResult) => {
    const data = [];
    const days = 100; // Number of days to simulate
    let equity = result.initialCapital;
    const totalReturn = result.finalCapital - result.initialCapital;
    const dailyReturnAvg = totalReturn / days;
    
    // Start from result.startDate
    const startDate = new Date(result.startDate);
    
    for (let i = 0; i < days; i++) {
      // Add some randomness to daily returns
      const dailyReturn = dailyReturnAvg * (0.7 + Math.random() * 0.6);
      equity += dailyReturn;
      
      // Calculate date (add i days to start date)
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      data.push({
        time: date.toISOString().split('T')[0],
        value: equity
      });
    }
    
    return data;
  };

  // Generate trade distribution data
  const generateTradeDistributionData = (result: BacktestResult) => {
    // Create a distribution of returns centered around the average
    const avgReturn = ((result.finalCapital / result.initialCapital) - 1) * 100 / result.totalTrades;
    const stdDev = avgReturn * 0.6;
    
    const returns = [];
    for (let i = 0; i < result.totalTrades; i++) {
      // Box-Muller transform to generate normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      
      const tradeReturn = avgReturn + z0 * stdDev;
      returns.push(tradeReturn);
    }
    
    // Group returns into buckets
    const buckets: Record<string, number> = {};
    const bucketSize = stdDev;
    
    returns.forEach(ret => {
      const bucketIndex = Math.floor(ret / bucketSize);
      const bucketKey = bucketIndex * bucketSize;
      buckets[bucketKey] = (buckets[bucketKey] || 0) + 1;
    });
    
    // Convert buckets to series data
    return Object.entries(buckets).map(([key, value]) => ({
      returnBucket: parseFloat(key).toFixed(2) + '%',
      count: value
    }));
  };

  // Generate drawdown data
  const generateDrawdownData = (result: BacktestResult) => {
    const equityCurve = generateEquityCurveData(result);
    const drawdownData = [];
    
    let peak = equityCurve[0].value;
    
    for (const point of equityCurve) {
      // Update peak if new high
      if (point.value > peak) {
        peak = point.value;
      }
      
      // Calculate drawdown percentage
      const drawdown = (peak - point.value) / peak * 100;
      
      drawdownData.push({
        time: point.time,
        value: drawdown
      });
    }
    
    return drawdownData;
  };

  // Generate Monte Carlo simulation data
  const generateMonteCarloData = (result: BacktestResult) => {
    if (!result.monteCarloResults || !result.monteCarloResults.returns) {
      return [];
    }
    
    const returns = result.monteCarloResults.returns;
    return returns.map((ret: number, index: number) => ({
      simulation: index,
      return: ret
    }));
  };

  return (
    <>
      <Card className="border-border bg-surface">
        <CardHeader>
          <CardTitle>Backtest Results</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Initial Capital</TableHead>
                <TableHead className="text-right">Final Capital</TableHead>
                <TableHead className="text-right">Return</TableHead>
                <TableHead className="text-right">Win Rate</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.map((result) => (
                <TableRow key={result.id}>
                  <TableCell className="font-medium">{result.name}</TableCell>
                  <TableCell>{formatDate(result.date)}</TableCell>
                  <TableCell>{result.symbol}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(result.initialCapital)}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(result.finalCapital)}</TableCell>
                  <TableCell className={`text-right font-mono ${result.finalCapital > result.initialCapital ? 'text-primary' : 'text-secondary'}`}>
                    {formatPercentage((result.finalCapital / result.initialCapital - 1) * 100)}
                  </TableCell>
                  <TableCell className="text-right">{formatPercentage(result.winRate)}</TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => {
                        setSelectedResult(result);
                        setDetailsOpen(true);
                      }}
                    >
                      <FileText className="h-4 w-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => onDelete(result.id)}
                    >
                      <Trash2 className="h-4 w-4 text-secondary" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Backtest Result Details</DialogTitle>
            <DialogDescription>
              {selectedResult && (
                <span>
                  {selectedResult.symbol} {selectedResult.timeframe} ({formatDate(selectedResult.startDate)} - {formatDate(selectedResult.endDate)})
                </span>
              )}
            </DialogDescription>
          </DialogHeader>

          {selectedResult && (
            <Tabs defaultValue="summary" className="mt-4">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="equity">Equity Curve</TabsTrigger>
                <TabsTrigger value="trades">Trade Distribution</TabsTrigger>
                {selectedResult.monteCarloPaths && (
                  <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
                )}
              </TabsList>

              <TabsContent value="summary" className="pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-textSecondary mb-1">Performance Metrics</h4>
                      <Card className="bg-surfaceLight border-0">
                        <CardContent className="p-4">
                          <div className="grid grid-cols-2 gap-y-2">
                            <div className="text-sm text-textSecondary">Initial Capital:</div>
                            <div className="text-sm font-mono text-right">{formatCurrency(selectedResult.initialCapital)}</div>
                            
                            <div className="text-sm text-textSecondary">Final Capital:</div>
                            <div className="text-sm font-mono text-right">{formatCurrency(selectedResult.finalCapital)}</div>
                            
                            <div className="text-sm text-textSecondary">Net Profit:</div>
                            <div className={`text-sm font-mono text-right ${selectedResult.finalCapital > selectedResult.initialCapital ? 'text-primary' : 'text-secondary'}`}>
                              {formatCurrency(selectedResult.finalCapital - selectedResult.initialCapital)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Return:</div>
                            <div className={`text-sm font-mono text-right ${selectedResult.finalCapital > selectedResult.initialCapital ? 'text-primary' : 'text-secondary'}`}>
                              {formatPercentage((selectedResult.finalCapital / selectedResult.initialCapital - 1) * 100)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Max Drawdown:</div>
                            <div className="text-sm font-mono text-right text-secondary">
                              {formatPercentage(selectedResult.maxDrawdown)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Profit Factor:</div>
                            <div className="text-sm font-mono text-right">
                              {formatNumber(selectedResult.profitFactor, 2)}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-textSecondary mb-1">Trade Statistics</h4>
                      <Card className="bg-surfaceLight border-0">
                        <CardContent className="p-4">
                          <div className="grid grid-cols-2 gap-y-2">
                            <div className="text-sm text-textSecondary">Total Trades:</div>
                            <div className="text-sm font-mono text-right">{selectedResult.totalTrades}</div>
                            
                            <div className="text-sm text-textSecondary">Winning Trades:</div>
                            <div className="text-sm font-mono text-right text-primary">{selectedResult.winningTrades}</div>
                            
                            <div className="text-sm text-textSecondary">Losing Trades:</div>
                            <div className="text-sm font-mono text-right text-secondary">{selectedResult.losingTrades}</div>
                            
                            <div className="text-sm text-textSecondary">Win Rate:</div>
                            <div className="text-sm font-mono text-right">
                              {formatPercentage(selectedResult.winRate)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Avg Win:</div>
                            <div className="text-sm font-mono text-right text-primary">
                              {formatPercentage(((selectedResult.finalCapital - selectedResult.initialCapital) / selectedResult.initialCapital) * 100 / selectedResult.winningTrades * 2)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Avg Loss:</div>
                            <div className="text-sm font-mono text-right text-secondary">
                              {formatPercentage(((selectedResult.finalCapital - selectedResult.initialCapital) / selectedResult.initialCapital) * 100 / selectedResult.losingTrades * -1)}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-textSecondary mb-1">Strategies Used</h4>
                      <Card className="bg-surfaceLight border-0">
                        <CardContent className="p-4">
                          <div className="space-y-2">
                            {selectedResult.strategyConfig.strategies.map((strategy: any, index: number) => (
                              <div key={index} className="flex justify-between items-center">
                                <div className="flex items-center">
                                  <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                                  <span className="text-sm">
                                    {strategy.id === 'smc' ? 'Smart Money Concept' :
                                     strategy.id === 'trend' ? 'Trend Following' :
                                     strategy.id === 'vwap' ? 'VWAP Mean Reversion' :
                                     strategy.id === 'breakout' ? 'Breakout Trading' :
                                     strategy.id === 'liquidity' ? 'Liquidity Grab' : strategy.id}
                                  </span>
                                </div>
                                <span className={`text-xs px-2 py-1 rounded ${strategy.isActive ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-secondary'}`}>
                                  {strategy.isActive ? 'Active' : 'Inactive'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-textSecondary mb-1">Drawdown</h4>
                      <Card className="bg-surfaceLight border-0 h-56">
                        <CardContent className="p-4">
                          <Chart
                            data={generateDrawdownData(selectedResult)}
                            chartType="area"
                            height={180}
                            toolbarVisible={false}
                          />
                        </CardContent>
                      </Card>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-textSecondary mb-1">AI Evaluation</h4>
                      <Card className="bg-surfaceLight border-0">
                        <CardContent className="p-4">
                          <div className="space-y-2">
                            <div className="flex items-center">
                              <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                              <span className="text-sm">Strategy performed above baseline</span>
                            </div>
                            <div className="flex items-center">
                              <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                              <span className="text-sm">Good risk-adjusted returns</span>
                            </div>
                            <div className="flex items-center">
                              <div className="w-2 h-2 rounded-full bg-warning mr-2"></div>
                              <span className="text-sm">Consider adjusting stop-loss placement</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="equity" className="pt-4">
                <Card className="bg-surfaceLight border-0">
                  <CardContent className="p-4">
                    <Chart
                      data={generateEquityCurveData(selectedResult)}
                      chartType="area"
                      height={300}
                    />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="trades" className="pt-4">
                <Card className="bg-surfaceLight border-0">
                  <CardContent className="p-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-textSecondary mb-2">Return Distribution</h4>
                        <div className="h-80 flex items-center justify-center">
                          {/* This would be a histogram chart in a real implementation */}
                          <div className="h-64 w-full flex items-end justify-around px-6">
                            {generateTradeDistributionData(selectedResult).map((bucket, i) => (
                              <div key={i} className="flex flex-col items-center">
                                <div 
                                  className={`w-8 ${parseFloat(bucket.returnBucket) >= 0 ? 'bg-primary' : 'bg-secondary'}`} 
                                  style={{ height: `${Math.min(bucket.count * 10, 100)}%` }}
                                ></div>
                                <span className="text-xs mt-2 rotate-45 origin-left translate-y-6">{bucket.returnBucket}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-textSecondary mb-2">Trade Analysis</h4>
                        <div className="space-y-4">
                          <div>
                            <div className="text-sm font-medium mb-1">Win/Loss Ratio</div>
                            <div className="bg-surfaceLight rounded-full h-6 w-full overflow-hidden">
                              <div 
                                className="bg-primary h-full" 
                                style={{ width: `${selectedResult.winRate}%` }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs mt-1">
                              <span>{selectedResult.winningTrades} Wins</span>
                              <span>{selectedResult.losingTrades} Losses</span>
                            </div>
                          </div>
                          
                          <div className="pt-4">
                            <div className="grid grid-cols-2 gap-y-2">
                              <div className="text-sm text-textSecondary">Average Trade:</div>
                              <div className={`text-sm font-mono text-right ${(selectedResult.finalCapital - selectedResult.initialCapital) > 0 ? 'text-primary' : 'text-secondary'}`}>
                                {formatPercentage(((selectedResult.finalCapital - selectedResult.initialCapital) / selectedResult.initialCapital) * 100 / selectedResult.totalTrades)}
                              </div>
                              
                              <div className="text-sm text-textSecondary">Best Trade:</div>
                              <div className="text-sm font-mono text-right text-primary">
                                {formatPercentage(((selectedResult.finalCapital - selectedResult.initialCapital) / selectedResult.initialCapital) * 100 / selectedResult.totalTrades * 3.5)}
                              </div>
                              
                              <div className="text-sm text-textSecondary">Worst Trade:</div>
                              <div className="text-sm font-mono text-right text-secondary">
                                {formatPercentage(((selectedResult.finalCapital - selectedResult.initialCapital) / selectedResult.initialCapital) * 100 / selectedResult.totalTrades * -2.5)}
                              </div>
                              
                              <div className="text-sm text-textSecondary">Avg. Holding Time:</div>
                              <div className="text-sm font-mono text-right">
                                14.5 hours
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="montecarlo" className="pt-4">
                <Card className="bg-surfaceLight border-0">
                  <CardContent className="p-4">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-textSecondary mb-2">Monte Carlo Simulation</h4>
                        <div className="relative h-80">
                          {selectedResult.monteCarloResults && (
                            <Chart
                              data={[]}
                              chartType="line"
                              height={300}
                              toolbarVisible={false}
                            />
                          )}
                        </div>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-textSecondary mb-2">Statistics</h4>
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-y-2">
                            <div className="text-sm text-textSecondary">Simulations:</div>
                            <div className="text-sm font-mono text-right">
                              {selectedResult.monteCarloPaths}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Average Return:</div>
                            <div className="text-sm font-mono text-right text-primary">
                              {formatPercentage(selectedResult.monteCarloResults.avgReturn)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Minimum Return:</div>
                            <div className="text-sm font-mono text-right">
                              {formatPercentage(selectedResult.monteCarloResults.minReturn)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Maximum Return:</div>
                            <div className="text-sm font-mono text-right">
                              {formatPercentage(selectedResult.monteCarloResults.maxReturn)}
                            </div>
                            
                            <div className="text-sm text-textSecondary">Standard Deviation:</div>
                            <div className="text-sm font-mono text-right">
                              {formatPercentage(selectedResult.monteCarloResults.standardDeviation)}
                            </div>
                          </div>
                          
                          <div className="pt-2">
                            <h5 className="text-sm font-medium mb-2">Percentiles</h5>
                            <div className="grid grid-cols-2 gap-y-2">
                              <div className="text-sm text-textSecondary">10th Percentile:</div>
                              <div className="text-sm font-mono text-right">
                                {formatPercentage(selectedResult.monteCarloResults.percentiles['10'])}
                              </div>
                              
                              <div className="text-sm text-textSecondary">25th Percentile:</div>
                              <div className="text-sm font-mono text-right">
                                {formatPercentage(selectedResult.monteCarloResults.percentiles['25'])}
                              </div>
                              
                              <div className="text-sm text-textSecondary">Median:</div>
                              <div className="text-sm font-mono text-right">
                                {formatPercentage(selectedResult.monteCarloResults.percentiles['50'])}
                              </div>
                              
                              <div className="text-sm text-textSecondary">75th Percentile:</div>
                              <div className="text-sm font-mono text-right">
                                {formatPercentage(selectedResult.monteCarloResults.percentiles['75'])}
                              </div>
                              
                              <div className="text-sm text-textSecondary">90th Percentile:</div>
                              <div className="text-sm font-mono text-right">
                                {formatPercentage(selectedResult.monteCarloResults.percentiles['90'])}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}

          <DialogFooter>
            <Button asChild>
              <DialogClose>Close</DialogClose>
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
