import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_ENDPOINTS } from "@/lib/constants";

interface HeaderProps {
  toggleSidebar: () => void;
}

export function Header({ toggleSidebar }: HeaderProps) {
  const [currentTime, setCurrentTime] = useState(new Date());

  // Connection status query
  const { data: connectionStatus } = useQuery({
    queryKey: [`${API_ENDPOINTS.SYSTEM}/connection-status`],
    refetchInterval: 30000, // Check every 30 seconds
  });

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);
  
  // Format the current time
  const formattedTime = currentTime.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
  
  const formattedDate = currentTime.toLocaleDateString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
  
  return (
    <header className="bg-background border-b border-border h-16 flex items-center justify-between px-4">
      <div className="flex items-center">
        <button 
          onClick={toggleSidebar}
          className="mr-4 text-2xl md:hidden text-muted-foreground hover:text-foreground transition-colors"
        >
          <i className="ri-menu-line"></i>
        </button>
        
        <div className="hidden md:flex items-center space-x-6">
          <div className="flex items-center">
            <span className="text-sm text-muted-foreground">{formattedDate}</span>
            <span className="mx-2 text-muted-foreground">|</span>
            <span className="text-sm font-medium">{formattedTime}</span>
          </div>
          
          <div className="h-5 border-r border-border"></div>
          
          <div className="flex items-center">
            <span 
              className={`inline-block w-2 h-2 rounded-full mr-2 ${connectionStatus?.isConnected ? 'bg-success' : 'bg-secondary'}`}
            ></span>
            <span className="text-sm">
              {connectionStatus?.isConnected ? 'Binance Connected' : 'Binance Disconnected'}
            </span>
          </div>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <button className="text-muted-foreground hover:text-foreground transition-colors">
          <i className="ri-notification-3-line text-xl"></i>
        </button>
        
        <button className="text-muted-foreground hover:text-foreground transition-colors">
          <i className="ri-user-line text-xl"></i>
        </button>
      </div>
    </header>
  );
}