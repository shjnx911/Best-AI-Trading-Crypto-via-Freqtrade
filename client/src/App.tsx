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
import { Header } from "@/components/ui/header";
import { useState } from "react";

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-background text-textPrimary font-sans">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
      <main className="flex-1 overflow-y-auto">
        <Header toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
        
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
