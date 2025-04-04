import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { SYSTEM_STATUSES } from "@/lib/constants";
import { useQuery } from "@tanstack/react-query";
import { API_ENDPOINTS } from "@/lib/constants";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const [location] = useLocation();
  
  const { data: systemStatus } = useQuery({
    queryKey: [`${API_ENDPOINTS.SYSTEM}/status`],
    staleTime: 30000, // 30 seconds
  });

  // Navigation items
  const navItems = [
    { href: "/", icon: "ri-dashboard-line", label: "Dashboard" },
    { href: "/bot-settings", icon: "ri-settings-5-line", label: "Bot Settings" },
    { href: "/backtesting", icon: "ri-time-line", label: "Backtesting" },
    { href: "/data-management", icon: "ri-database-2-line", label: "Data Management" },
    { href: "/trading-history", icon: "ri-line-chart-line", label: "Trading History" },
    { href: "/ai-training", icon: "ri-ai-generate", label: "AI Training" },
    { href: "/system-settings", icon: "ri-settings-3-line", label: "System Settings" },
  ];

  // Calculate display class based on isOpen prop
  const displayClass = isOpen 
    ? "block fixed inset-y-0 left-0 z-50 w-64 md:relative md:block"
    : "hidden md:block md:relative";

  // Handle click outside on mobile
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden" 
          onClick={handleOverlayClick}
        />
      )}
      
      <aside className={cn(
        "w-64 shrink-0 bg-surface border-r border-border",
        displayClass
      )}>
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-border">
            <div className="flex items-center justify-between">
              <h1 className="text-xl font-bold text-textPrimary flex items-center">
                <i className="ri-robot-2-line mr-2 text-primary"></i>
                BinanceAI Trader
              </h1>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-primary rounded-full"></span>
                <span className="ml-1 text-xs text-primary">Online</span>
              </div>
            </div>
          </div>
          
          <nav className="flex-1 overflow-y-auto py-4">
            <ul className="space-y-1 px-2">
              {navItems.map((item) => (
                <li key={item.href}>
                  <Link href={item.href}>
                    <a className={cn(
                      "flex items-center p-2 text-base font-medium rounded-md transition",
                      location === item.href
                        ? "text-primary bg-primary/10"
                        : "text-textSecondary hover:text-textPrimary hover:bg-surfaceLight"
                    )}>
                      <i className={cn(item.icon, "mr-2")}></i>
                      {item.label}
                    </a>
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
          
          <div className="p-4 border-t border-border">
            <div className="p-3 bg-surfaceLight rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">System Status</span>
                <div className="flex items-center">
                  <span className={cn(
                    "w-2 h-2 rounded-full",
                    systemStatus?.status === SYSTEM_STATUSES.ONLINE 
                      ? "bg-primary" 
                      : systemStatus?.status === SYSTEM_STATUSES.MAINTENANCE 
                        ? "bg-warning" 
                        : "bg-secondary"
                  )}></span>
                  <span className={cn(
                    "ml-1 text-xs",
                    systemStatus?.status === SYSTEM_STATUSES.ONLINE 
                      ? "text-primary" 
                      : systemStatus?.status === SYSTEM_STATUSES.MAINTENANCE 
                        ? "text-warning" 
                        : "text-secondary"
                  )}>
                    {systemStatus?.status === SYSTEM_STATUSES.ONLINE 
                      ? "Active" 
                      : systemStatus?.status === SYSTEM_STATUSES.MAINTENANCE 
                        ? "Maintenance" 
                        : "Offline"}
                  </span>
                </div>
              </div>
              
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-textSecondary">CPU</span>
                <span className="text-xs">{systemStatus?.cpu ?? 0}%</span>
              </div>
              <div className="w-full bg-background rounded-full h-1.5 mb-2">
                <div 
                  className="bg-primary h-1.5 rounded-full" 
                  style={{ width: `${systemStatus?.cpu ?? 0}%` }}
                ></div>
              </div>
              
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-textSecondary">RAM</span>
                <span className="text-xs">{systemStatus?.ram ?? 0}%</span>
              </div>
              <div className="w-full bg-background rounded-full h-1.5 mb-2">
                <div 
                  className="bg-accent h-1.5 rounded-full" 
                  style={{ width: `${systemStatus?.ram ?? 0}%` }}
                ></div>
              </div>
              
              <div className="flex justify-between text-xs text-textSecondary mt-2">
                <span>{systemStatus?.systemInfo?.cpu ?? "CPU"}</span>
                <span>{systemStatus?.systemInfo?.ram ?? "RAM"}</span>
                <span>{systemStatus?.systemInfo?.gpu ?? "GPU"}</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
