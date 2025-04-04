import { AccountSummary } from '@/components/account-summary';
import { PositionsTable } from '@/components/positions-table';
import { MarketAnalysis } from '@/components/market-analysis';
import { BotStatus } from '@/components/bot-status';
import { RecentTradesTable } from '@/components/recent-trades-table';

export default function Dashboard() {
  return (
    <div className="p-4 lg:p-6">
      {/* Account Summary */}
      <AccountSummary />
      
      {/* Active Positions */}
      <PositionsTable />
      
      {/* Market Analysis and Bot Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <MarketAnalysis />
        <BotStatus />
      </div>
      
      {/* Recent Trades */}
      <RecentTradesTable />
    </div>
  );
}
