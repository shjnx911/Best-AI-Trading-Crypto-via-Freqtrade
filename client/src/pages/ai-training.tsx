import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Chart } from '@/components/ui/chart';
import { fetchAIModelStats, fetchAIInsights, trainAIModel, pauseAITraining, resumeAITraining, getTrainingProgress } from '@/lib/ai-model';
import { ArrowRight, Pause, Play, AlertTriangle, Brain } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function AITraining() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [trainingParams, setTrainingParams] = useState({
    datasets: ['price_data', 'indicators', 'market_sentiment'],
    epochs: 50,
    batchSize: 32
  });

  // Fetch AI model stats
  const { data: modelStats, isLoading: isLoadingStats } = useQuery({
    queryKey: [`${API_ENDPOINTS.AI}/stats`],
    queryFn: fetchAIModelStats,
  });

  // Fetch AI insights
  const { data: insights, isLoading: isLoadingInsights } = useQuery({
    queryKey: [`${API_ENDPOINTS.AI}/insights`],
    queryFn: fetchAIInsights,
  });

  // Fetch training progress
  const { data: trainingProgress, isLoading: isLoadingProgress } = useQuery({
    queryKey: [`${API_ENDPOINTS.AI}/training-progress`],
    queryFn: getTrainingProgress,
    refetchInterval: (data) => (data && 'isTraining' in data && data.isTraining) ? 2000 : false,
  });

  // Train model mutation
  const trainModelMutation = useMutation({
    mutationFn: trainAIModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.AI}/training-progress`] });
      toast({
        title: "Training started",
        description: "AI model training has been started successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to start training: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Pause training mutation
  const pauseTrainingMutation = useMutation({
    mutationFn: pauseAITraining,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.AI}/training-progress`] });
      toast({
        title: "Training paused",
        description: "AI model training has been paused",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to pause training: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Resume training mutation
  const resumeTrainingMutation = useMutation({
    mutationFn: resumeAITraining,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.AI}/training-progress`] });
      toast({
        title: "Training resumed",
        description: "AI model training has been resumed",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to resume training: ${error}`,
        variant: "destructive",
      });
    },
  });

  const handleTrainModel = () => {
    trainModelMutation.mutate(trainingParams);
  };

  const handlePauseTraining = () => {
    pauseTrainingMutation.mutate();
  };

  const handleResumeTraining = () => {
    resumeTrainingMutation.mutate();
  };

  const updateTrainingParam = (key: string, value: any) => {
    setTrainingParams({ ...trainingParams, [key]: value });
  };

  // Generate mock training history data
  const generateTrainingHistoryData = () => {
    const data = [];
    const days = 30;
    let accuracy = 50;
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      
      // Gradually increase accuracy with some randomness
      accuracy += Math.random() * 1.2 - 0.3;
      if (accuracy > 85) accuracy = 85; // Cap at 85%
      
      data.push({
        date: date.toISOString().split('T')[0],
        accuracy
      });
    }
    
    return data;
  };

  return (
    <div className="p-4 lg:p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">AI Training</h1>
        <div className="flex items-center space-x-2">
          {trainingProgress?.isTraining ? (
            <div className="flex items-center space-x-2">
              <div className="text-sm text-textSecondary">Training in progress: {Math.round(trainingProgress.progress)}%</div>
              <Button 
                variant="outline"
                onClick={handlePauseTraining}
                disabled={pauseTrainingMutation.isPending}
              >
                <Pause className="h-4 w-4 mr-2" />
                Pause
              </Button>
            </div>
          ) : (
            trainingProgress?.progress > 0 && trainingProgress?.progress < 100 ? (
              <Button 
                variant="outline"
                onClick={handleResumeTraining}
                disabled={resumeTrainingMutation.isPending}
              >
                <Play className="h-4 w-4 mr-2" />
                Resume
              </Button>
            ) : null
          )}
        </div>
      </div>

      <Tabs defaultValue="training">
        <TabsList className="mb-6">
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="training" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <Card className="border-border bg-surface">
                <CardHeader>
                  <CardTitle>Model Training</CardTitle>
                  <CardDescription>Configure and run AI model training</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="epochs" className="mb-2 block">Training Epochs</Label>
                    <Input 
                      id="epochs" 
                      type="number" 
                      value={trainingParams.epochs}
                      onChange={(e) => updateTrainingParam('epochs', parseInt(e.target.value))}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Number of complete passes through the training dataset</p>
                  </div>

                  <div>
                    <Label htmlFor="batch-size" className="mb-2 block">Batch Size</Label>
                    <Input 
                      id="batch-size" 
                      type="number" 
                      value={trainingParams.batchSize}
                      onChange={(e) => updateTrainingParam('batchSize', parseInt(e.target.value))}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Number of samples processed before model update</p>
                  </div>

                  <div>
                    <Label className="mb-2 block">Training Datasets</Label>
                    <div className="space-y-2 bg-surfaceLight p-3 rounded-md">
                      <div className="flex items-center">
                        <input 
                          type="checkbox" 
                          id="dataset-price" 
                          checked={trainingParams.datasets.includes('price_data')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              updateTrainingParam('datasets', [...trainingParams.datasets, 'price_data']);
                            } else {
                              updateTrainingParam('datasets', trainingParams.datasets.filter(d => d !== 'price_data'));
                            }
                          }}
                          className="mr-2"
                        />
                        <Label htmlFor="dataset-price">Price Data</Label>
                      </div>
                      <div className="flex items-center">
                        <input 
                          type="checkbox" 
                          id="dataset-indicators" 
                          checked={trainingParams.datasets.includes('indicators')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              updateTrainingParam('datasets', [...trainingParams.datasets, 'indicators']);
                            } else {
                              updateTrainingParam('datasets', trainingParams.datasets.filter(d => d !== 'indicators'));
                            }
                          }}
                          className="mr-2"
                        />
                        <Label htmlFor="dataset-indicators">Technical Indicators</Label>
                      </div>
                      <div className="flex items-center">
                        <input 
                          type="checkbox" 
                          id="dataset-sentiment" 
                          checked={trainingParams.datasets.includes('market_sentiment')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              updateTrainingParam('datasets', [...trainingParams.datasets, 'market_sentiment']);
                            } else {
                              updateTrainingParam('datasets', trainingParams.datasets.filter(d => d !== 'market_sentiment'));
                            }
                          }}
                          className="mr-2"
                        />
                        <Label htmlFor="dataset-sentiment">Market Sentiment</Label>
                      </div>
                      <div className="flex items-center">
                        <input 
                          type="checkbox" 
                          id="dataset-trades" 
                          checked={trainingParams.datasets.includes('trading_history')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              updateTrainingParam('datasets', [...trainingParams.datasets, 'trading_history']);
                            } else {
                              updateTrainingParam('datasets', trainingParams.datasets.filter(d => d !== 'trading_history'));
                            }
                          }}
                          className="mr-2"
                        />
                        <Label htmlFor="dataset-trades">Trading History</Label>
                      </div>
                    </div>
                  </div>

                  {trainingProgress?.isTraining ? (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Training Progress</span>
                        <span>{Math.round(trainingProgress.progress)}%</span>
                      </div>
                      <Progress value={trainingProgress.progress} className="h-2" />
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-textSecondary">Epoch:</span> {trainingProgress.currentEpoch}/{trainingProgress.totalEpochs}
                        </div>
                        <div>
                          <span className="text-textSecondary">Accuracy:</span> {trainingProgress.accuracy.toFixed(2)}%
                        </div>
                        <div>
                          <span className="text-textSecondary">Loss:</span> {trainingProgress.loss.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Button 
                      className="w-full bg-primary hover:bg-primary/90" 
                      onClick={handleTrainModel}
                      disabled={trainModelMutation.isPending}
                    >
                      {trainModelMutation.isPending ? 'Starting Training...' : 'Start Training'}
                    </Button>
                  )}
                </CardContent>
              </Card>
            </div>

            <div className="lg:col-span-2">
              <Card className="border-border bg-surface">
                <CardHeader>
                  <CardTitle>Model Information</CardTitle>
                </CardHeader>
                <CardContent>
                  {isLoadingStats ? (
                    <div className="animate-pulse space-y-4">
                      <div className="h-6 bg-surfaceLight rounded w-1/4"></div>
                      <div className="h-20 bg-surfaceLight rounded"></div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <Card className="bg-surfaceLight border-0">
                          <CardContent className="p-4">
                            <div className="flex items-center mb-2">
                              <Brain className="h-5 w-5 text-primary mr-2" />
                              <h3 className="text-lg font-medium">AI Model</h3>
                            </div>
                            <div className="space-y-2 mt-4">
                              <div className="flex justify-between">
                                <span className="text-textSecondary">Version:</span>
                                <span>{modelStats?.version}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-textSecondary">Last Trained:</span>
                                <span>{modelStats?.lastTraining}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-textSecondary">Training Data Points:</span>
                                <span>{modelStats?.trainingData.toLocaleString()}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-textSecondary">Improvement:</span>
                                <span className="text-primary">+{modelStats?.accuracyImprovement}%</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      <div>
                        <h3 className="text-sm font-medium text-textSecondary mb-3">Training History</h3>
                        <div className="h-64">
                          <Chart
                            data={generateTrainingHistoryData()}
                            chartType="line"
                            height={250}
                            xKey="date"
                            yKey="accuracy"
                            yName="Accuracy (%)"
                            color="var(--chart-1)"
                            toolbarVisible={false}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="insights">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>AI Insights</CardTitle>
              <CardDescription>Insights and recommendations from the AI model</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingInsights ? (
                <div className="animate-pulse space-y-4">
                  <div className="h-20 bg-surfaceLight rounded"></div>
                  <div className="h-20 bg-surfaceLight rounded"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  {insights?.map((insight, index) => (
                    <Card 
                      key={index} 
                      className={`bg-surfaceLight border-0 ${
                        insight.type === 'success' ? 'border-l-4 border-l-primary' : 
                        insight.type === 'warning' ? 'border-l-4 border-l-warning' : 
                        'border-l-4 border-l-secondary'
                      }`}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start">
                          {insight.type === 'success' ? (
                            <div className="h-6 w-6 rounded-full bg-primary/10 flex items-center justify-center text-primary mr-3 mt-0.5">
                              <i className="ri-check-line"></i>
                            </div>
                          ) : insight.type === 'warning' ? (
                            <div className="h-6 w-6 rounded-full bg-warning/10 flex items-center justify-center text-warning mr-3 mt-0.5">
                              <AlertTriangle className="h-3.5 w-3.5" />
                            </div>
                          ) : (
                            <div className="h-6 w-6 rounded-full bg-secondary/10 flex items-center justify-center text-secondary mr-3 mt-0.5">
                              <i className="ri-close-line"></i>
                            </div>
                          )}
                          <div>
                            <p className="text-sm">{insight.message}</p>
                            {insight.details && (
                              <p className="text-xs text-textSecondary mt-1">{insight.details}</p>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
              <CardDescription>Analyze AI model performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium text-textSecondary mb-3">Strategy Performance Comparison</h3>
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border text-textSecondary text-sm">
                        <th className="text-left py-2">Strategy</th>
                        <th className="text-right py-2">Win Rate</th>
                        <th className="text-right py-2">AI Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-border">
                        <td className="py-3">Smart Money Concept</td>
                        <td className="text-right text-primary">72.4%</td>
                        <td className="text-right">85</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">AI Trend Following</td>
                        <td className="text-right text-primary">68.9%</td>
                        <td className="text-right">82</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">VWAP Mean Reversion</td>
                        <td className="text-right text-primary">65.2%</td>
                        <td className="text-right">78</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">Breakout Trading</td>
                        <td className="text-right text-primary">61.5%</td>
                        <td className="text-right">75</td>
                      </tr>
                      <tr>
                        <td className="py-3">Liquidity Grab</td>
                        <td className="text-right text-primary">58.7%</td>
                        <td className="text-right">72</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-textSecondary mb-3">Timeframe Effectiveness</h3>
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-border text-textSecondary text-sm">
                        <th className="text-left py-2">Timeframe</th>
                        <th className="text-right py-2">Accuracy</th>
                        <th className="text-right py-2">Trades</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-border">
                        <td className="py-3">5 minute</td>
                        <td className="text-right">62.3%</td>
                        <td className="text-right">843</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">15 minute</td>
                        <td className="text-right">65.7%</td>
                        <td className="text-right">612</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">1 hour</td>
                        <td className="text-right text-primary">74.2%</td>
                        <td className="text-right">278</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-3">4 hour</td>
                        <td className="text-right text-primary">78.5%</td>
                        <td className="text-right">145</td>
                      </tr>
                      <tr>
                        <td className="py-3">1 day</td>
                        <td className="text-right">68.9%</td>
                        <td className="text-right">65</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-sm font-medium text-textSecondary mb-3">Symbol Performance</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="bg-surfaceLight border-0">
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">BTC/USDT</span>
                        <span className="text-primary">76.3%</span>
                      </div>
                      <Progress value={76.3} className="h-1.5" />
                    </CardContent>
                  </Card>
                  <Card className="bg-surfaceLight border-0">
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">ETH/USDT</span>
                        <span className="text-primary">72.1%</span>
                      </div>
                      <Progress value={72.1} className="h-1.5" />
                    </CardContent>
                  </Card>
                  <Card className="bg-surfaceLight border-0">
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">SOL/USDT</span>
                        <span className="text-primary">68.5%</span>
                      </div>
                      <Progress value={68.5} className="h-1.5" />
                    </CardContent>
                  </Card>
                  <Card className="bg-surfaceLight border-0">
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">ADA/USDT</span>
                        <span>65.2%</span>
                      </div>
                      <Progress value={65.2} className="h-1.5" />
                    </CardContent>
                  </Card>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}