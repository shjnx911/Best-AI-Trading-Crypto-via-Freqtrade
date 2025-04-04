import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchTradingHistory } from '@/lib/binance';
import { formatCurrency, formatDateTime, formatNumber } from '@/lib/utils';
import { TradeHistory } from '@shared/types';
import { Pagination } from '@/components/ui/pagination';

export function RecentTradesTable() {
  const [page, setPage] = useState(1);
  const pageSize = 10;

  // Fetch trading history
  const { data, isLoading } = useQuery({
    queryKey: [`${API_ENDPOINTS.TRADES}`, page, pageSize],
    queryFn: () => fetchTradingHistory(page, pageSize),
  });

  const trades = data?.trades || [];
  const totalTrades = data?.total || 0;
  const totalPages = Math.ceil(totalTrades / pageSize);

  if (isLoading) {
    return (
      <div className="bg-surface rounded-lg border border-border p-4">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surfaceLight rounded w-1/4"></div>
          <div className="h-64 bg-surfaceLight rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Recent Trades</h2>
        <div>
          <button className="bg-surfaceLight rounded-md px-3 py-1.5 text-sm hover:bg-surfaceLight/80">
            <i className="ri-history-line mr-1"></i>
            View All History
          </button>
        </div>
      </div>
      
      <div className="bg-surface rounded-lg border border-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-xs text-textSecondary">
                <th className="px-4 py-3 text-left font-medium border-b border-border">Date & Time</th>
                <th className="px-4 py-3 text-left font-medium border-b border-border">Symbol</th>
                <th className="px-4 py-3 text-left font-medium border-b border-border">Side</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Size</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Entry Price</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Exit Price</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">PnL</th>
                <th className="px-4 py-3 text-left font-medium border-b border-border">Strategy</th>
                <th className="px-4 py-3 text-center font-medium border-b border-border">AI Score</th>
              </tr>
            </thead>
            <tbody>
              {trades.length > 0 ? (
                trades.map((trade: TradeHistory) => (
                  <tr key={trade.id} className="border-b border-border hover:bg-surfaceLight/40 text-sm">
                    <td className="px-4 py-3 text-textSecondary">{formatDateTime(trade.timestamp)}</td>
                    <td className="px-4 py-3 font-medium">{trade.symbol}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        trade.side === 'LONG' ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-secondary'
                      }`}>
                        {trade.side}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {formatNumber(trade.size)} {trade.symbol.split('/')[0]}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(trade.entryPrice)}</td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(trade.exitPrice)}</td>
                    <td className={`px-4 py-3 text-right font-mono ${
                      trade.pnl > 0 ? 'text-primary' : 'text-secondary'
                    }`}>
                      {trade.pnl > 0 ? '+' : ''}
                      {formatCurrency(trade.pnl)} ({trade.pnlPercent > 0 ? '+' : ''}
                      {formatNumber(trade.pnlPercent)}%)
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-xs">{trade.strategy}</span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <div className={`inline-flex items-center px-2 py-1 rounded ${
                        trade.aiScore >= 70 ? 'bg-primary/10 text-primary' : 
                        trade.aiScore >= 50 ? 'bg-warning/10 text-warning' : 
                        'bg-secondary/10 text-secondary'
                      }`}>
                        <i className="ri-ai-generate mr-1 text-xs"></i>
                        <span>{trade.aiScore}%</span>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-textSecondary">
                    No trading history found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        {totalPages > 1 && (
          <div className="flex justify-center py-4">
            <Pagination>
              <Pagination.Prev 
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
              />
              
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                // Always show first page, last page, current page and 1 page on each side of current
                const pageNumbers = [];
                if (totalPages <= 5) {
                  // Show all pages if 5 or fewer
                  for (let i = 1; i <= totalPages; i++) {
                    pageNumbers.push(i);
                  }
                } else {
                  // Always show first page
                  pageNumbers.push(1);
                  
                  // Calculate range around current page
                  const startPage = Math.max(2, page - 1);
                  const endPage = Math.min(totalPages - 1, page + 1);
                  
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
                      isActive={pageNum === page}
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
      </div>
    </div>
  );
}
