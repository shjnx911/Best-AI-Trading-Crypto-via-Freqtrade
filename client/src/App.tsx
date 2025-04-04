import { Route, Switch } from "wouter";
import Dashboard from "@/pages/dashboard";
import BotSettings from "@/pages/bot-settings";
import Backtesting from "@/pages/backtesting";
import DataManagement from "@/pages/data-management";
import TradingHistory from "@/pages/trading-history";
import AITraining from "@/pages/ai-training";
import SystemSettings from "@/pages/system-settings";
import NotFound from "@/pages/not-found";
import { Sidebar } from "@/components/ui/sidebar";
import { useState } from "react";

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-background text-textPrimary font-sans">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
      <main className="flex-1 overflow-y-auto">
        <header className="h-16 bg-surface border-b border-border px-4 flex items-center justify-between">
          <div className="flex items-center">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
              className="md:hidden text-textPrimary mr-2"
            >
              <i className="ri-menu-line text-xl"></i>
            </button>
            <div className="ml-2 flex items-center">
              <div className="bg-surfaceLight text-textSecondary rounded-md px-3 py-1.5 text-sm font-medium flex items-center">
                <i className="ri-exchange-funds-line mr-1"></i>
                <span className="text-warning">Testnet</span>
              </div>
              <div className="ml-2 bg-surfaceLight text-textSecondary rounded-md px-3 py-1.5 text-sm font-medium flex items-center">
                <i className="ri-ai-generate mr-1"></i>
                <span>AI Status: <span className="text-primary">Learning</span></span>
              </div>
            </div>
          </div>
          <div className="flex items-center">
            <button className="relative text-textSecondary hover:text-textPrimary p-2">
              <i className="ri-notification-3-line text-xl"></i>
              <span className="absolute top-1 right-1 w-2 h-2 bg-primary rounded-full"></span>
            </button>
            <div className="ml-4 flex items-center border border-border rounded-md overflow-hidden">
              <div className="px-3 py-1.5 bg-surfaceLight text-xs text-textSecondary">Total Balance</div>
              <div className="px-3 py-1.5 font-mono font-medium">$8,245.63 USDT</div>
            </div>
            <div className="ml-2">
              <button className="flex items-center space-x-2 bg-surfaceLight rounded-md px-3 py-1.5 text-sm hover:bg-surfaceLight/80">
                <div className="w-6 h-6 rounded-full bg-accent flex items-center justify-center text-white">
                  <i className="ri-user-line"></i>
                </div>
                <span className="hidden sm:inline">User</span>
                <i className="ri-arrow-down-s-line"></i>
              </button>
            </div>
          </div>
        </header>
        
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/bot-settings" component={BotSettings} />
          <Route path="/backtesting" component={Backtesting} />
          <Route path="/data-management" component={DataManagement} />
          <Route path="/trading-history" component={TradingHistory} />
          <Route path="/ai-training" component={AITraining} />
          <Route path="/system-settings" component={SystemSettings} />
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

export default App;
