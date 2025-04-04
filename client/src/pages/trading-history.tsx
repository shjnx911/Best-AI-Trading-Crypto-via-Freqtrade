import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchTradingHistory } from '@/lib/binance';
import { TradeHistory } from '@shared/types';
import { analyzeTradePerformance } from '@/lib/ai-model';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Pagination } from '@/components/ui/pagination';
import { formatCurrency, formatDateTime, formatNumber, formatPercentage } from '@/lib/utils';
import { Chart } from '@/components/ui/chart';
import { FileText, ChartBar, ChevronDown, AlertCircle } from 'lucide-react';

export default function TradingHistory() {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedTrade, setSelectedTrade] = useState<TradeHistory | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [dateRange, setDateRange] = useState('all');
  const [sortBy, setSortBy] = useState('date-desc');
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [selectedTradesForAnalysis, setSelectedTradesForAnalysis] = useState<number[]>([]);

  // Fetch trading history
  const { data, isLoading } = useQuery({
    queryKey: [`${API_ENDPOINTS.TRADES}`, page, pageSize],
    queryFn: () => fetchTradingHistory(page, pageSize),
  });

  // Fetch trade analysis
  const { data: tradeAnalysis, isLoading: isLoadingAnalysis } = useQuery({
    queryKey: [`${API_ENDPOINTS.AI}/analyze-trades`, selectedTradesForAnalysis],
    queryFn: () => analyzeTradePerformance(selectedTradesForAnalysis),
    enabled: analysisOpen && selectedTradesForAnalysis.length > 0,
  });

  const trades = data?.trades || [];
  const totalTrades = data?.total || 0;
  const totalPages = Math.ceil(totalTrades / pageSize);

  // Generate mock equity curve data from trade history
  const generateEquityCurveData = () => {
    if (!trades || trades.length === 0) return [];

    const sortedTrades = [...trades].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    
    let equity = 10000; // Start with arbitrary equity
    const data = [{ time: new Date(sortedTrades[0].timestamp).toISOString().split('T')[0], value: equity }];
    
    sortedTrades.forEach(trade => {
      equity += trade.pnl;
      data.push({
        time: new Date(trade.timestamp).toISOString().split('T')[0],
        value: equity
      });
    });
    
    return data;
  };

  // Calculate trading statistics
  const calculateStatistics = () => {
    if (!trades || trades.length === 0) return {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalProfit: 0,
      avgProfit: 0,
      largestWin: 0,
      largestLoss: 0,
      profitFactor: 0
    };

    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    
    const totalProfit = trades.reduce((sum, t) => sum + t.pnl, 0);
    const avgProfit = totalProfit / trades.length;
    
    const largestWin = winningTrades.length > 0 
      ? Math.max(...winningTrades.map(t => t.pnl))
      : 0;
      
    const largestLoss = losingTrades.length > 0 
      ? Math.min(...losingTrades.map(t => t.pnl))
      : 0;
    
    const totalWinAmount = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const totalLossAmount = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
    
    const profitFactor = totalLossAmount !== 0 ? totalWinAmount / totalLossAmount : 0;
    
    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: (winningTrades.length / trades.length) * 100,
      totalProfit,
      avgProfit,
      largestWin,
      largestLoss,
      profitFactor
    };
  };

  const statistics = calculateStatistics();

  const handleOpenDetails = (trade: TradeHistory) => {
    setSelectedTrade(trade);
    setDetailsOpen(true);
  };

  const handleAnalyzeSelected = () => {
    // In a real app, we would get the IDs from selected trades
    // For now, let's use the first 5 trade IDs
    const tradeIds = trades.slice(0, 5).map(t => t.id);
    setSelectedTradesForAnalysis(tradeIds);
    setAnalysisOpen(true);
  };

  // Filter trades based on selected filter
  const getFilteredTrades = () => {
    if (selectedFilter === 'all') return trades;
    if (selectedFilter === 'winning') return trades.filter(t => t.pnl > 0);
    if (selectedFilter === 'losing') return trades.filter(t => t.pnl < 0);
    
    // Filter by strategy
    return trades.filter(t => t.strategy.toLowerCase().includes(selectedFilter));
  };

  const filteredTrades = getFilteredTrades();

  if (isLoading) {
    return (
      <div className="p-4 lg:p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surfaceLight rounded w-1/4"></div>
          <div className="h-64 bg-surfaceLight rounded"></div>
          <div className="h-64 bg-surfaceLight rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Trading History</h1>
        <div className="flex space-x-2">
          <Button onClick={handleAnalyzeSelected} className="bg-accent text-white hover:bg-accent/90">
            <ChartBar className="h-4 w-4 mr-2" />
            Analyze Performance
          </Button>
        </div>
      </div>

      <Tabs defaultValue="trades">
        <TabsList className="mb-6">
          <TabsTrigger value="trades">Trade History</TabsTrigger>
          <TabsTrigger value="statistics">Performance Statistics</TabsTrigger>
          <TabsTrigger value="equity">Equity Curve</TabsTrigger>
        </TabsList>

        <TabsContent value="trades">
          <Card className="border-border bg-surface">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Trade History</CardTitle>
                <div className="flex items-center space-x-2">
                  <Select value={selectedFilter} onValueChange={setSelectedFilter}>
                    <SelectTrigger className="w-[150px] bg-surfaceLight">
                      <SelectValue placeholder="Filter" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Trades</SelectItem>
                      <SelectItem value="winning">Winning Trades</SelectItem>
                      <SelectItem value="losing">Losing Trades</SelectItem>
                      <SelectItem value="smc">Smart Money Concept</SelectItem>
                      <SelectItem value="trend">Trend Following</SelectItem>
                      <SelectItem value="vwap">VWAP Mean Reversion</SelectItem>
                      <SelectItem value="breakout">Breakout</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select value={dateRange} onValueChange={setDateRange}>
                    <SelectTrigger className="w-[150px] bg-surfaceLight">
                      <SelectValue placeholder="Date Range" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Time</SelectItem>
                      <SelectItem value="today">Today</SelectItem>
                      <SelectItem value="week">This Week</SelectItem>
                      <SelectItem value="month">This Month</SelectItem>
                      <SelectItem value="custom">Custom Range</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select value={sortBy} onValueChange={setSortBy}>
                    <SelectTrigger className="w-[150px] bg-surfaceLight">
                      <SelectValue placeholder="Sort By" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="date-desc">Date (Newest)</SelectItem>
                      <SelectItem value="date-asc">Date (Oldest)</SelectItem>
                      <SelectItem value="pnl-desc">PnL (Highest)</SelectItem>
                      <SelectItem value="pnl-asc">PnL (Lowest)</SelectItem>
                      <SelectItem value="ai-desc">AI Score (Highest)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="border border-border rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">Date & Time</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead className="text-right">Size</TableHead>
                      <TableHead className="text-right">Entry Price</TableHead>
                      <TableHead className="text-right">Exit Price</TableHead>
                      <TableHead className="text-right">PnL</TableHead>
                      <TableHead>Strategy</TableHead>
                      <TableHead className="text-center">AI Score</TableHead>
                      <TableHead className="w-[80px] text-right">Details</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredTrades.length > 0 ? (
                      filteredTrades.map((trade) => (
                        <TableRow key={trade.id}>
                          <TableCell className="text-textSecondary">{formatDateTime(trade.timestamp)}</TableCell>
                          <TableCell className="font-medium">{trade.symbol}</TableCell>
                          <TableCell>
                            <span className={`px-2 py-0.5 rounded text-xs ${
                              trade.side === 'LONG' ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-secondary'
                            }`}>
                              {trade.side}
                            </span>
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {formatNumber(trade.size)} {trade.symbol.split('/')[0]}
                          </TableCell>
                          <TableCell className="text-right font-mono">{formatCurrency(trade.entryPrice)}</TableCell>
                          <TableCell className="text-right font-mono">{formatCurrency(trade.exitPrice)}</TableCell>
                          <TableCell className={`text-right font-mono ${
                            trade.pnl > 0 ? 'text-primary' : 'text-secondary'
                          }`}>
                            {trade.pnl > 0 ? '+' : ''}
                            {formatCurrency(trade.pnl)} ({trade.pnlPercent > 0 ? '+' : ''}
                            {formatNumber(trade.pnlPercent)}%)
                          </TableCell>
                          <TableCell>
                            <span className="text-xs">{trade.strategy}</span>
                          </TableCell>
                          <TableCell className="text-center">
                            <div className={`inline-flex items-center px-2 py-1 rounded ${
                              trade.aiScore >= 70 ? 'bg-primary/10 text-primary' : 
                              trade.aiScore >= 50 ? 'bg-warning/10 text-warning' : 
                              'bg-secondary/10 text-secondary'
                            }`}>
                              <i className="ri-ai-generate mr-1 text-xs"></i>
                              <span>{trade.aiScore}%</span>
                            </div>
                          </TableCell>
                          <TableCell className="text-right">
                            <Button 
                              variant="ghost" 
                              size="icon"
                              onClick={() => handleOpenDetails(trade)}
                            >
                              <FileText className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={10} className="h-24 text-center">
                          No trades found.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>

              {totalPages > 1 && (
                <div className="flex justify-center mt-4">
                  <Pagination>
                    <Pagination.Previous 
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                    />
                    
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      const currentPage = page;
                      let pageNumbers = [];
                      
                      if (totalPages <= 5) {
                        // Show all pages if 5 or fewer
                        pageNumbers = Array.from({ length: totalPages }, (_, i) => i + 1);
                      } else {
                        // Always show first page
                        pageNumbers.push(1);
                        
                        // Calculate range around current page
                        const startPage = Math.max(2, currentPage - 1);
                        const endPage = Math.min(totalPages - 1, currentPage + 1);
                        
                        // Add ellipsis after first page if there's a gap
                        if (startPage > 2) {
                          pageNumbers.push(-1); // -1 represents ellipsis
                        }
                        
                        // Add pages around current page
                        for (let i = startPage; i <= endPage; i++) {
                          pageNumbers.push(i);
                        }
                        
                        // Add ellipsis before last page if there's a gap
                        if (endPage < totalPages - 1) {
                          pageNumbers.push(-2); // -2 represents ellipsis
                        }
                        
                        // Always show last page
                        pageNumbers.push(totalPages);
                      }
                      
                      return pageNumbers.map((pageNum, idx) => {
                        if (pageNum < 0) {
                          return (
                            <Pagination.Ellipsis key={`ellipsis-${pageNum}-${idx}`} />
                          );
                        }
                        
                        return (
                          <Pagination.Item
                            key={pageNum}
                            onClick={() => setPage(pageNum)}
                            isActive={pageNum === currentPage}
                          >
                            {pageNum}
                          </Pagination.Item>
                        );
                      });
                    })}
                    
                    <Pagination.Next 
                      onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                    />
                  </Pagination>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="statistics">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>Performance Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-6">
                    <div className="text-center p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Total Trades</div>
                      <div className="text-2xl font-semibold">{statistics.totalTrades}</div>
                    </div>
                    <div className="text-center p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Win Rate</div>
                      <div className="text-2xl font-semibold text-primary">{formatPercentage(statistics.winRate)}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-6">
                    <div className="text-center p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Winning Trades</div>
                      <div className="text-2xl font-semibold text-primary">{statistics.winningTrades}</div>
                    </div>
                    <div className="text-center p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Losing Trades</div>
                      <div className="text-2xl font-semibold text-secondary">{statistics.losingTrades}</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-surfaceLight rounded-lg">
                    <div className="text-sm text-textSecondary">Total Profit/Loss</div>
                    <div className={`text-xl font-semibold ${statistics.totalProfit > 0 ? 'text-primary' : 'text-secondary'}`}>
                      {statistics.totalProfit > 0 ? '+' : ''}
                      {formatCurrency(statistics.totalProfit)}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-6">
                    <div className="p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Average Profit</div>
                      <div className={`text-lg font-semibold ${statistics.avgProfit > 0 ? 'text-primary' : 'text-secondary'}`}>
                        {statistics.avgProfit > 0 ? '+' : ''}
                        {formatCurrency(statistics.avgProfit)}
                      </div>
                    </div>
                    <div className="p-4 bg-surfaceLight rounded-lg">
                      <div className="text-sm text-textSecondary mb-1">Profit Factor</div>
                      <div className="text-lg font-semibold">{statistics.profitFactor.toFixed(2)}</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>Performance by Strategy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Mock strategy performance data */}
                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">Smart Money Concept</div>
                      <div className="text-primary">72% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-primary h-full rounded-full" style={{ width: '72%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>18 trades</span>
                      <span className="text-primary">+$325.40</span>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">Trend Following</div>
                      <div className="text-primary">68% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-primary h-full rounded-full" style={{ width: '68%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>22 trades</span>
                      <span className="text-primary">+$412.60</span>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">VWAP Mean Reversion</div>
                      <div className="text-primary">64% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-primary h-full rounded-full" style={{ width: '64%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>14 trades</span>
                      <span className="text-primary">+$198.20</span>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">Breakout Trading</div>
                      <div className="text-secondary">48% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-secondary h-full rounded-full" style={{ width: '48%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>8 trades</span>
                      <span className="text-secondary">-$85.30</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>Performance by Symbol</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Mock symbol performance data */}
                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">BTC/USDT</div>
                      <div className="text-primary">78% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-primary h-full rounded-full" style={{ width: '78%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>23 trades</span>
                      <span className="text-primary">+$482.90</span>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">ETH/USDT</div>
                      <div className="text-primary">65% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-primary h-full rounded-full" style={{ width: '65%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>20 trades</span>
                      <span className="text-primary">+$298.40</span>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="font-medium">SOL/USDT</div>
                      <div className="text-secondary">55% Win Rate</div>
                    </div>
                    <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                      <div className="bg-accent h-full rounded-full" style={{ width: '55%' }}></div>
                    </div>
                    <div className="text-sm flex justify-between">
                      <span>12 trades</span>
                      <span className="text-primary">+$102.80</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>AI Performance Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-surfaceLight rounded-lg">
                    <h4 className="font-medium mb-2">Trade Success by AI Score</h4>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>AI Score 80-100%</span>
                          <span className="text-primary">87% Win Rate</span>
                        </div>
                        <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                          <div className="bg-primary h-full rounded-full" style={{ width: '87%' }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>AI Score 60-79%</span>
                          <span className="text-primary">72% Win Rate</span>
                        </div>
                        <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                          <div className="bg-primary h-full rounded-full" style={{ width: '72%' }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>AI Score 40-59%</span>
                          <span className="text-accent">59% Win Rate</span>
                        </div>
                        <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                          <div className="bg-accent h-full rounded-full" style={{ width: '59%' }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>AI Score 0-39%</span>
                          <span className="text-secondary">32% Win Rate</span>
                        </div>
                        <div className="w-full bg-background h-2 rounded-full overflow-hidden">
                          <div className="bg-secondary h-full rounded-full" style={{ width: '32%' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-surfaceLight rounded-lg">
                    <h4 className="font-medium mb-3">AI Recommendations</h4>
                    <ul className="space-y-2">
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">Favor high AI confidence trades (80%+) for optimal results</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">Smart Money Concept strategy performing best with current market conditions</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-warning rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">Consider adjusting trailing stop settings to improve overall win rate</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-warning rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">Optimal trade duration appears to be between 4-8 hours for current strategy mix</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="equity">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <Chart
                  data={generateEquityCurveData()}
                  chartType="area"
                  height={300}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Trade Details Dialog */}
      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Trade Details</DialogTitle>
          </DialogHeader>

          {selectedTrade && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="space-y-4">
                  <div className="bg-surfaceLight p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-3">Trade Information</h3>
                    <div className="grid grid-cols-2 gap-y-2">
                      <div className="text-sm text-textSecondary">Symbol:</div>
                      <div className="text-sm font-medium">{selectedTrade.symbol}</div>

                      <div className="text-sm text-textSecondary">Side:</div>
                      <div className={`text-sm font-medium ${
                        selectedTrade.side === 'LONG' ? 'text-primary' : 'text-secondary'
                      }`}>
                        {selectedTrade.side}
                      </div>

                      <div className="text-sm text-textSecondary">Open Time:</div>
                      <div className="text-sm">{formatDateTime(selectedTrade.timestamp)}</div>

                      <div className="text-sm text-textSecondary">Close Time:</div>
                      <div className="text-sm">{formatDateTime(new Date(new Date(selectedTrade.timestamp).getTime() + 1000 * 60 * 60 * 4))}</div>

                      <div className="text-sm text-textSecondary">Duration:</div>
                      <div className="text-sm">4h 12m</div>

                      <div className="text-sm text-textSecondary">Size:</div>
                      <div className="text-sm font-mono">{formatNumber(selectedTrade.size)} {selectedTrade.symbol.split('/')[0]}</div>

                      <div className="text-sm text-textSecondary">Leverage:</div>
                      <div className="text-sm font-mono">3x</div>
                    </div>
                  </div>

                  <div className="bg-surfaceLight p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-3">Performance</h3>
                    <div className="grid grid-cols-2 gap-y-2">
                      <div className="text-sm text-textSecondary">Entry Price:</div>
                      <div className="text-sm font-mono">{formatCurrency(selectedTrade.entryPrice)}</div>

                      <div className="text-sm text-textSecondary">Exit Price:</div>
                      <div className="text-sm font-mono">{formatCurrency(selectedTrade.exitPrice)}</div>

                      <div className="text-sm text-textSecondary">Profit/Loss:</div>
                      <div className={`text-sm font-mono ${
                        selectedTrade.pnl > 0 ? 'text-primary' : 'text-secondary'
                      }`}>
                        {selectedTrade.pnl > 0 ? '+' : ''}
                        {formatCurrency(selectedTrade.pnl)}
                      </div>

                      <div className="text-sm text-textSecondary">Return:</div>
                      <div className={`text-sm font-mono ${
                        selectedTrade.pnlPercent > 0 ? 'text-primary' : 'text-secondary'
                      }`}>
                        {selectedTrade.pnlPercent > 0 ? '+' : ''}
                        {formatNumber(selectedTrade.pnlPercent)}%
                      </div>

                      <div className="text-sm text-textSecondary">ROI (with leverage):</div>
                      <div className={`text-sm font-mono ${
                        selectedTrade.pnlPercent * 3 > 0 ? 'text-primary' : 'text-secondary'
                      }`}>
                        {selectedTrade.pnlPercent * 3 > 0 ? '+' : ''}
                        {formatNumber(selectedTrade.pnlPercent * 3)}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <div className="space-y-4">
                  <div className="bg-surfaceLight p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-3">Strategy</h3>
                    <div className="grid grid-cols-2 gap-y-2">
                      <div className="text-sm text-textSecondary">Strategy:</div>
                      <div className="text-sm font-medium">{selectedTrade.strategy}</div>

                      <div className="text-sm text-textSecondary">Entry Signal:</div>
                      <div className="text-sm">{selectedTrade.side === 'LONG' ? 'Bullish Breakout' : 'Bearish Breakdown'}</div>

                      <div className="text-sm text-textSecondary">Exit Signal:</div>
                      <div className="text-sm">Take Profit Hit</div>

                      <div className="text-sm text-textSecondary">AI Score:</div>
                      <div className={`text-sm ${
                        selectedTrade.aiScore >= 70 ? 'text-primary' : 
                        selectedTrade.aiScore >= 50 ? 'text-warning' : 
                        'text-secondary'
                      }`}>
                        {selectedTrade.aiScore}%
                      </div>
                    </div>

                    <div className="mt-4">
                      <div className="text-sm text-textSecondary mb-1">AI Notes:</div>
                      <div className="text-sm mt-1 bg-background p-2 rounded">
                        {selectedTrade.aiNotes || "No AI notes available for this trade."}
                      </div>
                    </div>
                  </div>

                  <div className="bg-surfaceLight p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-3">Market Conditions</h3>
                    <div className="grid grid-cols-2 gap-y-2">
                      <div className="text-sm text-textSecondary">Market Trend:</div>
                      <div className="text-sm">{selectedTrade.pnl > 0 ? 'Bullish' : 'Bearish'}</div>

                      <div className="text-sm text-textSecondary">Volatility:</div>
                      <div className="text-sm">Medium</div>

                      <div className="text-sm text-textSecondary">Volume:</div>
                      <div className="text-sm">Above Average</div>

                      <div className="text-sm text-textSecondary">Key Indicators:</div>
                      <div className="text-sm">
                        {selectedTrade.side === 'LONG' ? 'RSI: 62, MACD: Bullish' : 'RSI: 38, MACD: Bearish'}
                      </div>
                    </div>
                  </div>

                  <div className="bg-surfaceLight p-4 rounded-lg">
                    <h3 className="text-lg font-medium mb-2">Trade Chart</h3>
                    <div className="h-40 bg-background rounded-md flex items-center justify-center text-sm text-textSecondary">
                      Trade chart visualization would be displayed here
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* AI Analysis Dialog */}
      <Dialog open={analysisOpen} onOpenChange={setAnalysisOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>AI Performance Analysis</DialogTitle>
          </DialogHeader>

          {isLoadingAnalysis ? (
            <div className="h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="loading-indicator mb-4 mx-auto"></div>
                <p className="text-textSecondary">Analyzing trading performance...</p>
              </div>
            </div>
          ) : tradeAnalysis ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-surfaceLight p-4 rounded-lg">
                  <h3 className="text-lg font-medium mb-3">Trading Strengths</h3>
                  <ul className="space-y-2">
                    {tradeAnalysis.strengths.map((strength, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-surfaceLight p-4 rounded-lg">
                  <h3 className="text-lg font-medium mb-3">Trading Weaknesses</h3>
                  <ul className="space-y-2">
                    {tradeAnalysis.weaknesses.map((weakness, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-secondary rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">{weakness}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-surfaceLight p-4 rounded-lg">
                  <h3 className="text-lg font-medium mb-3">Recommendations</h3>
                  <ul className="space-y-2">
                    {tradeAnalysis.recommendations.map((recommendation, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-accent rounded-full mr-2 mt-1.5"></div>
                        <span className="text-sm">{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center">
              <div className="text-center">
                <AlertCircle className="h-12 w-12 text-textSecondary mb-3 mx-auto" />
                <p className="text-textSecondary">No trades selected for analysis.</p>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setAnalysisOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
