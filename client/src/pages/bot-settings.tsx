import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchBotSettings, updateBotSettings, toggleBot } from '@/lib/binance';
import { StrategySettings } from '@/components/strategy-settings';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { fetchAvailableSymbols } from '@/lib/binance';

export default function BotSettings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [availableSymbol, setAvailableSymbol] = useState<string>('BTC/USDT');

  // Fetch bot settings
  const { data: botSettings, isLoading: isLoadingSettings } = useQuery({
    queryKey: [`${API_ENDPOINTS.BOT}/settings`],
    queryFn: fetchBotSettings,
  });

  // Fetch available symbols
  const { data: availableSymbols, isLoading: isLoadingSymbols } = useQuery({
    queryKey: [`${API_ENDPOINTS.ACCOUNT}/symbols`],
    queryFn: fetchAvailableSymbols,
  });

  // Update bot settings mutation
  const updateSettingsMutation = useMutation({
    mutationFn: updateBotSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BOT}/settings`] });
      toast({
        title: "Settings updated",
        description: "Bot settings have been updated successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to update settings: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Toggle bot mutation
  const toggleBotMutation = useMutation({
    mutationFn: toggleBot,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BOT}/status`] });
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.BOT}/settings`] });
      toast({
        title: botSettings?.isActive ? "Bot deactivated" : "Bot activated",
        description: botSettings?.isActive 
          ? "The trading bot has been stopped" 
          : "The trading bot has been started",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to toggle bot status: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Update settings handler
  const handleUpdateSettings = (updatedSettings: any) => {
    updateSettingsMutation.mutate(updatedSettings);
  };

  // Update trading pairs
  useEffect(() => {
    if (botSettings) {
      const symbols = botSettings.tradingPairs?.map((pair: any) => pair.symbol) || [];
      setSelectedSymbols(symbols);
    }
  }, [botSettings]);

  // Add trading pair
  const handleAddSymbol = () => {
    if (availableSymbol && !selectedSymbols.includes(availableSymbol)) {
      const newSymbols = [...selectedSymbols, availableSymbol];
      setSelectedSymbols(newSymbols);
      
      // Update bot settings with new trading pairs
      const updatedSettings = {
        ...botSettings,
        tradingPairs: newSymbols.map(symbol => ({ symbol, isActive: true }))
      };
      handleUpdateSettings(updatedSettings);
    }
  };

  // Remove trading pair
  const handleRemoveSymbol = (symbolToRemove: string) => {
    const newSymbols = selectedSymbols.filter(symbol => symbol !== symbolToRemove);
    setSelectedSymbols(newSymbols);
    
    // Update bot settings with new trading pairs
    const updatedSettings = {
      ...botSettings,
      tradingPairs: newSymbols.map(symbol => ({ symbol, isActive: true }))
    };
    handleUpdateSettings(updatedSettings);
  };

  // Toggle bot status
  const handleToggleBot = () => {
    toggleBotMutation.mutate(!botSettings?.isActive);
  };

  if (isLoadingSettings) {
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
        <h1 className="text-2xl font-bold">Bot Settings</h1>
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <Switch 
              id="bot-active" 
              checked={botSettings?.isActive || false}
              onCheckedChange={handleToggleBot}
              className="mr-2"
            />
            <Label htmlFor="bot-active">
              {botSettings?.isActive ? 'Bot Active' : 'Bot Inactive'}
            </Label>
          </div>
          <Button 
            className="bg-primary hover:bg-primary/90" 
            onClick={() => handleUpdateSettings(botSettings)}
            disabled={updateSettingsMutation.isPending}
          >
            {updateSettingsMutation.isPending ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="trading">
        <TabsList className="mb-6">
          <TabsTrigger value="trading">Trading Configuration</TabsTrigger>
          <TabsTrigger value="strategies">Strategy Settings</TabsTrigger>
          <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="trading" className="space-y-6">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Trading Pairs</CardTitle>
              <CardDescription>Configure which cryptocurrency pairs the bot will trade</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-4 mb-6">
                <div className="flex-1">
                  <Label htmlFor="symbol-select" className="mb-2 block">Add Trading Pair</Label>
                  <Select value={availableSymbol} onValueChange={setAvailableSymbol}>
                    <SelectTrigger id="symbol-select" className="bg-surfaceLight">
                      <SelectValue placeholder="Select trading pair" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableSymbols?.map(symbol => (
                        <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={handleAddSymbol}>Add Pair</Button>
              </div>

              <div className="border border-border rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="bg-surfaceLight">
                      <th className="px-4 py-3 text-left text-sm font-medium text-textSecondary">Symbol</th>
                      <th className="px-4 py-3 text-right text-sm font-medium text-textSecondary">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedSymbols.length > 0 ? (
                      selectedSymbols.map(symbol => (
                        <tr key={symbol} className="border-t border-border">
                          <td className="px-4 py-3 text-left">{symbol}</td>
                          <td className="px-4 py-3 text-right">
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              className="text-secondary"
                              onClick={() => handleRemoveSymbol(symbol)}
                            >
                              <i className="ri-delete-bin-line mr-1"></i>
                              Remove
                            </Button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={2} className="px-4 py-6 text-center text-textSecondary">
                          No trading pairs added. Add a pair to start trading.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Capital Management</CardTitle>
              <CardDescription>Configure how much capital to use for each trade</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="max-pairs" className="mb-2 block">Maximum Active Pairs</Label>
                    <Input 
                      id="max-pairs" 
                      type="number" 
                      value={botSettings?.maxPairs || 3}
                      onChange={(e) => handleUpdateSettings({ 
                        ...botSettings, 
                        maxPairs: parseInt(e.target.value) 
                      })}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Maximum number of pairs to trade simultaneously</p>
                  </div>

                  <div>
                    <Label htmlFor="leverage" className="mb-2 block">Default Leverage</Label>
                    <Input 
                      id="leverage" 
                      type="number" 
                      value={botSettings?.defaultLeverage || 3}
                      onChange={(e) => handleUpdateSettings({ 
                        ...botSettings, 
                        defaultLeverage: parseInt(e.target.value) 
                      })}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Default leverage to use for trades</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between mb-2">
                    <Label>AI Position Sizing</Label>
                    <Switch 
                      checked={botSettings?.aiPositionSizingEnabled || false}
                      onCheckedChange={(checked) => handleUpdateSettings({
                        ...botSettings,
                        aiPositionSizingEnabled: checked
                      })}
                    />
                  </div>
                  <p className="text-sm text-textSecondary mb-4">
                    When enabled, AI will adjust position sizes based on confidence signals and market conditions.
                  </p>

                  <div className="flex items-center justify-between mb-2">
                    <Label>Auto Balance Capital</Label>
                    <Switch 
                      checked={botSettings?.autoBalanceCapital || false}
                      onCheckedChange={(checked) => handleUpdateSettings({
                        ...botSettings,
                        autoBalanceCapital: checked
                      })}
                    />
                  </div>
                  <p className="text-sm text-textSecondary">
                    Automatically balances capital across active trading pairs.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="strategies">
          <StrategySettings 
            strategyConfig={botSettings?.strategyConfig}
            onStrategyChange={(strategyConfig) => handleUpdateSettings({
              ...botSettings,
              strategyConfig
            })}
          />
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Advanced Trading Settings</CardTitle>
              <CardDescription>Configure advanced trading parameters</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="order-type" className="mb-2 block">Default Order Type</Label>
                    <Select 
                      value={botSettings?.orderType || "MARKET"}
                      onValueChange={(value) => handleUpdateSettings({
                        ...botSettings,
                        orderType: value
                      })}
                    >
                      <SelectTrigger id="order-type" className="bg-surfaceLight">
                        <SelectValue placeholder="Select order type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="MARKET">Market</SelectItem>
                        <SelectItem value="LIMIT">Limit</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-textSecondary mt-1">Default order type for entries and exits</p>
                  </div>

                  <div>
                    <Label htmlFor="entry-condition" className="mb-2 block">Entry Conditions</Label>
                    <Select 
                      value={botSettings?.entryCondition || "AGGRESSIVE"}
                      onValueChange={(value) => handleUpdateSettings({
                        ...botSettings,
                        entryCondition: value
                      })}
                    >
                      <SelectTrigger id="entry-condition" className="bg-surfaceLight">
                        <SelectValue placeholder="Select entry condition" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="CONSERVATIVE">Conservative</SelectItem>
                        <SelectItem value="MODERATE">Moderate</SelectItem>
                        <SelectItem value="AGGRESSIVE">Aggressive</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-textSecondary mt-1">How strict the bot should be with entry signals</p>
                  </div>

                  <div className="flex items-center justify-between mb-2">
                    <Label>Reduce Position on Loss</Label>
                    <Switch 
                      checked={botSettings?.reducePositionOnLoss || false}
                      onCheckedChange={(checked) => handleUpdateSettings({
                        ...botSettings,
                        reducePositionOnLoss: checked
                      })}
                    />
                  </div>
                  <p className="text-sm text-textSecondary">
                    Automatically reduce position size after consecutive losses.
                  </p>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="max-drawdown" className="mb-2 block">Maximum Daily Drawdown (%)</Label>
                    <Input 
                      id="max-drawdown" 
                      type="number" 
                      value={botSettings?.maxDailyDrawdown || 5}
                      onChange={(e) => handleUpdateSettings({ 
                        ...botSettings, 
                        maxDailyDrawdown: parseFloat(e.target.value) 
                      })}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Maximum allowed drawdown in a single day before stopping trading</p>
                  </div>

                  <div>
                    <Label htmlFor="max-trade-duration" className="mb-2 block">Maximum Trade Duration (hours)</Label>
                    <Input 
                      id="max-trade-duration" 
                      type="number" 
                      value={botSettings?.maxTradeDuration || 48}
                      onChange={(e) => handleUpdateSettings({ 
                        ...botSettings, 
                        maxTradeDuration: parseInt(e.target.value) 
                      })}
                      className="bg-surfaceLight"
                    />
                    <p className="text-xs text-textSecondary mt-1">Maximum time a position can be held before forced closure</p>
                  </div>

                  <div className="flex items-center justify-between mb-2">
                    <Label>Paper Trading After Target</Label>
                    <Switch 
                      checked={botSettings?.paperTradingAfterTarget || false}
                      onCheckedChange={(checked) => handleUpdateSettings({
                        ...botSettings,
                        paperTradingAfterTarget: checked
                      })}
                    />
                  </div>
                  <p className="text-sm text-textSecondary">
                    Switch to paper trading after daily profit target is reached.
                  </p>
                </div>
              </div>

              <Separator className="my-6" />

              <div>
                <h3 className="text-lg font-medium mb-4">Notifications</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between mb-2">
                      <Label>Trade Notifications</Label>
                      <Switch 
                        checked={botSettings?.tradeNotifications || true}
                        onCheckedChange={(checked) => handleUpdateSettings({
                          ...botSettings,
                          tradeNotifications: checked
                        })}
                      />
                    </div>
                    <p className="text-sm text-textSecondary">
                      Receive notifications for new trades, closures, and stop losses.
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between mb-2">
                      <Label>Performance Notifications</Label>
                      <Switch 
                        checked={botSettings?.performanceNotifications || true}
                        onCheckedChange={(checked) => handleUpdateSettings({
                          ...botSettings,
                          performanceNotifications: checked
                        })}
                      />
                    </div>
                    <p className="text-sm text-textSecondary">
                      Receive daily and weekly performance summaries.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
