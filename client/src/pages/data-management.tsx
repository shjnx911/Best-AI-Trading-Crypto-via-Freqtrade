import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Progress } from '@/components/ui/progress';
import { fetchAvailableSymbols } from '@/lib/binance';
import { TIMEFRAMES } from '@/lib/constants';
import { apiRequest } from '@/lib/queryClient';
import { formatDate } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { AlertCircle, DownloadCloud, Database, Trash2 } from 'lucide-react';

export default function DataManagement() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadParams, setDownloadParams] = useState({
    symbol: 'BTC/USDT',
    timeframe: '1h',
    startDate: '2022-01-01',
    endDate: new Date().toISOString().split('T')[0]
  });
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);

  // Fetch available symbols
  const { data: availableSymbols } = useQuery({
    queryKey: [`${API_ENDPOINTS.ACCOUNT}/symbols`],
    queryFn: fetchAvailableSymbols,
  });

  // Fetch available data
  const { data: availableData = [], isLoading: isLoadingData } = useQuery({
    queryKey: [`${API_ENDPOINTS.DATA}/available`],
    queryFn: async () => {
      const response = await apiRequest('GET', `${API_ENDPOINTS.DATA}/available`);
      return response.json();
    },
  });

  // Download data mutation
  const downloadDataMutation = useMutation({
    mutationFn: async (params: typeof downloadParams) => {
      setIsDownloading(true);
      setDownloadProgress(0);
      
      // Simulate progress
      const progressInterval = setInterval(() => {
        setDownloadProgress(prev => {
          const newProgress = prev + Math.random() * 10;
          return newProgress < 95 ? newProgress : 95;
        });
      }, 1000);
      
      try {
        const response = await apiRequest('POST', `${API_ENDPOINTS.DATA}/download`, params);
        clearInterval(progressInterval);
        setDownloadProgress(100);
        setTimeout(() => {
          setIsDownloading(false);
          setDownloadProgress(0);
        }, 1000);
        return response.json();
      } catch (error) {
        clearInterval(progressInterval);
        setIsDownloading(false);
        setDownloadProgress(0);
        throw error;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.DATA}/available`] });
      toast({
        title: "Data downloaded",
        description: "Market data has been downloaded successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to download data: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Delete data mutation
  const deleteDataMutation = useMutation({
    mutationFn: async ({ symbol, timeframe }: { symbol: string, timeframe: string }) => {
      const response = await apiRequest('DELETE', `${API_ENDPOINTS.DATA}/delete`, { symbol, timeframe });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.DATA}/available`] });
      toast({
        title: "Data deleted",
        description: "Market data has been deleted successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to delete data: ${error}`,
        variant: "destructive",
      });
    },
  });

  const handleDownloadData = () => {
    downloadDataMutation.mutate(downloadParams);
  };

  const handleDeleteData = (symbol: string, timeframe: string) => {
    if (confirm(`Are you sure you want to delete ${symbol} ${timeframe} data?`)) {
      deleteDataMutation.mutate({ symbol, timeframe });
    }
  };

  const updateParam = (key: string, value: any) => {
    setDownloadParams({ ...downloadParams, [key]: value });
  };

  // Calculate total data size
  const totalDataSize = availableData.reduce((total, item) => total + item.sizeInMB, 0);

  return (
    <div className="p-4 lg:p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Data Management</h1>
        <div className="flex items-center text-sm">
          <Database className="h-4 w-4 mr-2 text-primary" />
          <span>Total Data: {totalDataSize.toFixed(2)} MB</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card className="border-border bg-surface sticky top-4">
            <CardHeader>
              <CardTitle>Download Market Data</CardTitle>
              <CardDescription>Download historical data for backtesting</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="symbol" className="mb-2 block">Trading Pair</Label>
                <Select 
                  value={downloadParams.symbol} 
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
                  value={downloadParams.timeframe} 
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
                  value={downloadParams.startDate}
                  onChange={(e) => updateParam('startDate', e.target.value)}
                  className="bg-surfaceLight"
                />
              </div>

              <div>
                <Label htmlFor="end-date" className="mb-2 block">End Date</Label>
                <Input 
                  id="end-date" 
                  type="date" 
                  value={downloadParams.endDate}
                  onChange={(e) => updateParam('endDate', e.target.value)}
                  className="bg-surfaceLight"
                />
              </div>

              {isDownloading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Downloading...</span>
                    <span>{Math.round(downloadProgress)}%</span>
                  </div>
                  <Progress value={downloadProgress} className="h-2" />
                </div>
              )}

              <Button 
                className="w-full bg-primary hover:bg-primary/90" 
                onClick={handleDownloadData}
                disabled={isDownloading || downloadDataMutation.isPending}
              >
                {isDownloading ? (
                  <span className="flex items-center">
                    <DownloadCloud className="h-4 w-4 mr-2" />
                    Downloading...
                  </span>
                ) : (
                  <span className="flex items-center">
                    <DownloadCloud className="h-4 w-4 mr-2" />
                    Download Data
                  </span>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Available Data</CardTitle>
              <CardDescription>Historical price data available for backtesting</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingData ? (
                <div className="animate-pulse space-y-4">
                  <div className="h-8 bg-surfaceLight rounded w-full"></div>
                  <div className="h-8 bg-surfaceLight rounded w-full"></div>
                  <div className="h-8 bg-surfaceLight rounded w-full"></div>
                </div>
              ) : availableData.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Timeframe</TableHead>
                      <TableHead>Date Range</TableHead>
                      <TableHead>Candles</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {availableData.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{item.symbol}</TableCell>
                        <TableCell>{item.timeframe}</TableCell>
                        <TableCell>{formatDate(item.startDate)} - {formatDate(item.endDate)}</TableCell>
                        <TableCell>{item.candles.toLocaleString()}</TableCell>
                        <TableCell>{item.sizeInMB.toFixed(2)} MB</TableCell>
                        <TableCell className="text-right">
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            className="text-secondary"
                            onClick={() => handleDeleteData(item.symbol, item.timeframe)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-6">
                  <AlertCircle className="h-12 w-12 text-textSecondary mb-3" />
                  <h3 className="text-lg font-medium mb-2">No Data Available</h3>
                  <p className="text-sm text-textSecondary mb-4">
                    You haven't downloaded any market data yet. Download historical data to run backtests.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
