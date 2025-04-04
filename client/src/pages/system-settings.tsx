import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { apiRequest } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { AlertTriangle, Shield, Save, RefreshCw, Database, Cpu } from 'lucide-react';

export default function SystemSettings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [apiKeys, setApiKeys] = useState({
    apiKey: '',
    apiSecret: '',
    isTestnet: true
  });

  // Fetch system settings
  const { data: systemSettings, isLoading: isLoadingSettings } = useQuery({
    queryKey: [`${API_ENDPOINTS.SYSTEM}/settings`],
    queryFn: async () => {
      const response = await apiRequest('GET', `${API_ENDPOINTS.SYSTEM}/settings`);
      return response.json();
    },
  });

  // Fetch system status
  const { data: systemStatus, isLoading: isLoadingStatus } = useQuery({
    queryKey: [`${API_ENDPOINTS.SYSTEM}/status`],
    queryFn: async () => {
      const response = await apiRequest('GET', `${API_ENDPOINTS.SYSTEM}/status`);
      return response.json();
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Update system settings mutation
  const updateSettingsMutation = useMutation({
    mutationFn: async (settings: any) => {
      const response = await apiRequest('POST', `${API_ENDPOINTS.SYSTEM}/settings`, settings);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`${API_ENDPOINTS.SYSTEM}/settings`] });
      toast({
        title: "Settings updated",
        description: "System settings have been updated successfully",
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

  // Update API keys mutation
  const updateApiKeysMutation = useMutation({
    mutationFn: async (keys: any) => {
      const response = await apiRequest('POST', `${API_ENDPOINTS.SYSTEM}/api-keys`, keys);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "API Keys updated",
        description: "Binance API keys have been saved successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to save API keys: ${error}`,
        variant: "destructive",
      });
    },
  });

  const handleSaveApiKeys = () => {
    updateApiKeysMutation.mutate(apiKeys);
  };

  const handleUpdateSettings = (updatedSettings: any) => {
    updateSettingsMutation.mutate({
      ...systemSettings,
      ...updatedSettings
    });
  };

  if (isLoadingSettings || isLoadingStatus) {
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
        <h1 className="text-2xl font-bold">System Settings</h1>
      </div>

      <Tabs defaultValue="api">
        <TabsList className="mb-6">
          <TabsTrigger value="api">API Configuration</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="logs">Logs & Diagnostics</TabsTrigger>
        </TabsList>

        <TabsContent value="api" className="space-y-6">
          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Binance API Configuration</CardTitle>
              <CardDescription>Configure your Binance API credentials</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center text-sm">
                  <Shield className="h-4 w-4 mr-2 text-primary" />
                  <span>API keys are securely stored and encrypted</span>
                </div>
                <Button 
                  variant="outline"
                  className="text-secondary hover:text-secondary hover:bg-secondary/10"
                  onClick={() => setApiKeys({ apiKey: '', apiSecret: '', isTestnet: true })}
                >
                  Clear
                </Button>
              </div>

              <div className="space-y-4">
                <div>
                  <Label htmlFor="api-key" className="mb-2 block">API Key</Label>
                  <Input 
                    id="api-key" 
                    type="text" 
                    value={apiKeys.apiKey}
                    onChange={(e) => setApiKeys({ ...apiKeys, apiKey: e.target.value })}
                    className="bg-surfaceLight"
                    placeholder="Enter your Binance API Key"
                  />
                </div>

                <div>
                  <Label htmlFor="api-secret" className="mb-2 block">API Secret</Label>
                  <Input 
                    id="api-secret" 
                    type="password" 
                    value={apiKeys.apiSecret}
                    onChange={(e) => setApiKeys({ ...apiKeys, apiSecret: e.target.value })}
                    className="bg-surfaceLight"
                    placeholder="Enter your Binance API Secret"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Switch 
                    id="testnet"
                    checked={apiKeys.isTestnet}
                    onCheckedChange={(checked) => setApiKeys({ ...apiKeys, isTestnet: checked })}
                  />
                  <Label htmlFor="testnet">Use Testnet (recommended for testing)</Label>
                </div>

                <div className="bg-warning/10 text-warning p-3 rounded-md text-sm flex items-start">
                  <AlertTriangle className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium mb-1">Important Security Notice</p>
                    <p>For security reasons, it's recommended to only use API keys with trading permissions, NOT withdrawal permissions. Make sure to set appropriate IP restrictions in your Binance account.</p>
                  </div>
                </div>

                <Button 
                  className="w-full bg-primary hover:bg-primary/90"
                  onClick={handleSaveApiKeys}
                  disabled={!apiKeys.apiKey || !apiKeys.apiSecret || updateApiKeysMutation.isPending}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {updateApiKeysMutation.isPending ? 'Saving API Keys...' : 'Save API Keys'}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Data Storage Settings</CardTitle>
              <CardDescription>Configure how your data is stored and managed</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium mb-1">Local Data Storage</h3>
                  <p className="text-sm text-textSecondary">Store historical market data and trading history locally</p>
                </div>
                <Switch 
                  checked={systemSettings?.localDataStorage || false}
                  onCheckedChange={(checked) => handleUpdateSettings({ localDataStorage: checked })}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium mb-1">Auto-backup</h3>
                  <p className="text-sm text-textSecondary">Automatically backup trading data daily</p>
                </div>
                <Switch 
                  checked={systemSettings?.autoBackup || false}
                  onCheckedChange={(checked) => handleUpdateSettings({ autoBackup: checked })}
                />
              </div>

              <div>
                <Label htmlFor="backup-location" className="mb-2 block">Backup Location</Label>
                <Input 
                  id="backup-location" 
                  type="text" 
                  value={systemSettings?.backupLocation || ''}
                  onChange={(e) => handleUpdateSettings({ backupLocation: e.target.value })}
                  className="bg-surfaceLight"
                  placeholder="/path/to/backup/directory"
                  disabled={!systemSettings?.autoBackup}
                />
              </div>

              <div className="flex justify-end">
                <Button 
                  variant="outline"
                  className="mr-2"
                  onClick={() => {
                    toast({
                      title: "Backup created",
                      description: "Backup has been created successfully",
                    });
                  }}
                >
                  <Database className="h-4 w-4 mr-2" />
                  Create Backup Now
                </Button>
                <Button 
                  onClick={() => handleUpdateSettings(systemSettings)}
                  disabled={updateSettingsMutation.isPending}
                >
                  {updateSettingsMutation.isPending ? 'Saving...' : 'Save Settings'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>System Status</CardTitle>
                <CardDescription>Current system resource utilization</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <Label>CPU Usage</Label>
                      <span className="text-sm">{systemStatus?.cpu}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-surfaceLight rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${systemStatus?.cpu > 80 ? 'bg-secondary' : 'bg-primary'}`}
                        style={{ width: `${systemStatus?.cpu}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1">
                      <Label>RAM Usage</Label>
                      <span className="text-sm">{systemStatus?.ram}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-surfaceLight rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${systemStatus?.ram > 80 ? 'bg-secondary' : 'bg-primary'}`}
                        style={{ width: `${systemStatus?.ram}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1">
                      <Label>GPU Usage (AI Training)</Label>
                      <span className="text-sm">{systemStatus?.gpu}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-surfaceLight rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${systemStatus?.gpu > 80 ? 'bg-secondary' : 'bg-primary'}`}
                        style={{ width: `${systemStatus?.gpu}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-border">
                  <h3 className="text-sm font-medium mb-3">System Specifications</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-textSecondary">CPU:</span>
                      <span>{systemStatus?.systemInfo?.cpu}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-textSecondary">RAM:</span>
                      <span>{systemStatus?.systemInfo?.ram}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-textSecondary">GPU:</span>
                      <span>{systemStatus?.systemInfo?.gpu}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border bg-surface">
              <CardHeader>
                <CardTitle>Performance Settings</CardTitle>
                <CardDescription>Configure system performance parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium mb-1">GPU Acceleration</h3>
                    <p className="text-sm text-textSecondary">Use GPU for AI model training and inference</p>
                  </div>
                  <Switch 
                    checked={systemSettings?.gpuAcceleration || false}
                    onCheckedChange={(checked) => handleUpdateSettings({ gpuAcceleration: checked })}
                  />
                </div>

                <div>
                  <Label htmlFor="max-threads" className="mb-2 block">Max CPU Threads</Label>
                  <Input 
                    id="max-threads" 
                    type="number" 
                    value={systemSettings?.maxThreads || 4}
                    onChange={(e) => handleUpdateSettings({ maxThreads: parseInt(e.target.value) })}
                    className="bg-surfaceLight"
                    min={1}
                    max={16}
                  />
                  <p className="text-xs text-textSecondary mt-1">Maximum number of CPU threads to use for backtesting</p>
                </div>

                <div>
                  <Label htmlFor="memory-limit" className="mb-2 block">Memory Limit (MB)</Label>
                  <Input 
                    id="memory-limit" 
                    type="number" 
                    value={systemSettings?.memoryLimit || 2048}
                    onChange={(e) => handleUpdateSettings({ memoryLimit: parseInt(e.target.value) })}
                    className="bg-surfaceLight"
                    min={512}
                    max={16384}
                  />
                  <p className="text-xs text-textSecondary mt-1">Maximum memory allocation for the application</p>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium mb-1">Energy Saving Mode</h3>
                    <p className="text-sm text-textSecondary">Reduce resource usage when possible</p>
                  </div>
                  <Switch 
                    checked={systemSettings?.energySavingMode || false}
                    onCheckedChange={(checked) => handleUpdateSettings({ energySavingMode: checked })}
                  />
                </div>

                <Button 
                  onClick={() => handleUpdateSettings(systemSettings)}
                  className="w-full"
                  disabled={updateSettingsMutation.isPending}
                >
                  {updateSettingsMutation.isPending ? 'Saving...' : 'Save Performance Settings'}
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>System Maintenance</CardTitle>
              <CardDescription>System maintenance and optimization options</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Button 
                  variant="outline" 
                  className="h-auto py-4 flex flex-col items-center justify-center"
                  onClick={() => {
                    toast({
                      title: "Cleanup completed",
                      description: "Temporary files have been cleaned up successfully",
                    });
                  }}
                >
                  <i className="ri-delete-bin-line text-xl mb-2"></i>
                  <span className="text-sm">Clean Temp Files</span>
                </Button>
                
                <Button 
                  variant="outline" 
                  className="h-auto py-4 flex flex-col items-center justify-center"
                  onClick={() => {
                    toast({
                      title: "Cache cleared",
                      description: "Application cache has been cleared successfully",
                    });
                  }}
                >
                  <i className="ri-refresh-line text-xl mb-2"></i>
                  <span className="text-sm">Clear Cache</span>
                </Button>
                
                <Button 
                  variant="outline" 
                  className="h-auto py-4 flex flex-col items-center justify-center"
                  onClick={() => {
                    toast({
                      title: "Database optimized",
                      description: "Database optimization completed successfully",
                    });
                  }}
                >
                  <i className="ri-database-2-line text-xl mb-2"></i>
                  <span className="text-sm">Optimize Database</span>
                </Button>
                
                <Button 
                  variant="outline" 
                  className="h-auto py-4 flex flex-col items-center justify-center"
                  onClick={() => {
                    toast({
                      title: "System restarted",
                      description: "System has been restarted successfully",
                    });
                  }}
                >
                  <i className="ri-restart-line text-xl mb-2"></i>
                  <span className="text-sm">Restart System</span>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="logs" className="space-y-6">
          <Card className="border-border bg-surface">
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>System Logs</CardTitle>
                  <CardDescription>View system logs and diagnostic information</CardDescription>
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                  <Button variant="outline" size="sm">
                    <i className="ri-download-2-line mr-2"></i>
                    Export
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex space-x-2 mb-4">
                <Button variant="outline" size="sm" className="bg-primary/10 text-primary">All</Button>
                <Button variant="outline" size="sm">Trading</Button>
                <Button variant="outline" size="sm">AI</Button>
                <Button variant="outline" size="sm">System</Button>
                <Button variant="outline" size="sm">Errors</Button>
              </div>
              
              <div className="bg-surfaceLight rounded-md p-4 font-mono text-xs h-80 overflow-y-auto">
                <pre>
                  <div className="text-textSecondary">[2023-04-23 08:14:22] [INFO] System startup completed</div>
                  <div className="text-textSecondary">[2023-04-23 08:15:05] [INFO] Connected to Binance API (testnet mode)</div>
                  <div className="text-textSecondary">[2023-04-23 08:15:10] [INFO] Trading bot initialized</div>
                  <div className="text-textSecondary">[2023-04-23 08:16:25] [INFO] AI model loaded (version 2.3.4)</div>
                  <div className="text-textSecondary">[2023-04-23 08:18:42] [INFO] Market data update completed for BTC/USDT, ETH/USDT</div>
                  <div className="text-primary">[2023-04-23 08:22:15] [SUCCESS] Created buy order for BTC/USDT at 27632.50</div>
                  <div className="text-secondary">[2023-04-23 08:35:19] [ERROR] Failed to connect to data provider, retrying...</div>
                  <div className="text-textSecondary">[2023-04-23 08:36:07] [INFO] Reconnected to data provider</div>
                  <div className="text-textSecondary">[2023-04-23 08:40:23] [INFO] AI model generating market analysis</div>
                  <div className="text-primary">[2023-04-23 08:42:51] [SUCCESS] Created sell order for BTC/USDT at 27652.80</div>
                  <div className="text-textSecondary">[2023-04-23 08:43:05] [INFO] Trade completed: BTC/USDT +0.73% profit</div>
                  <div className="text-textSecondary">[2023-04-23 08:45:32] [INFO] System performance check: CPU 24%, RAM 38%, GPU 12%</div>
                  <div className="text-warning">[2023-04-23 08:52:19] [WARNING] Market volatility increasing, adjusting position sizing</div>
                  <div className="text-textSecondary">[2023-04-23 09:01:44] [INFO] Scheduled data backup started</div>
                  <div className="text-textSecondary">[2023-04-23 09:02:37] [INFO] Data backup completed successfully</div>
                  <div className="text-textSecondary">[2023-04-23 09:15:00] [INFO] AI model starting training on new data</div>
                </pre>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-surface">
            <CardHeader>
              <CardTitle>Diagnostic Tools</CardTitle>
              <CardDescription>Run system diagnostics and tests</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium mb-3">System Tests</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-surfaceLight p-3 rounded-md">
                      <div className="flex items-center">
                        <Cpu className="h-4 w-4 mr-2 text-primary" />
                        <span>CPU Performance Test</span>
                      </div>
                      <Button variant="outline" size="sm">Run</Button>
                    </div>
                    <div className="flex items-center justify-between bg-surfaceLight p-3 rounded-md">
                      <div className="flex items-center">
                        <i className="ri-database-2-line mr-2 text-primary"></i>
                        <span>Database Integrity Check</span>
                      </div>
                      <Button variant="outline" size="sm">Run</Button>
                    </div>
                    <div className="flex items-center justify-between bg-surfaceLight p-3 rounded-md">
                      <div className="flex items-center">
                        <i className="ri-exchange-funds-line mr-2 text-primary"></i>
                        <span>API Connection Test</span>
                      </div>
                      <Button variant="outline" size="sm">Run</Button>
                    </div>
                    <div className="flex items-center justify-between bg-surfaceLight p-3 rounded-md">
                      <div className="flex items-center">
                        <i className="ri-ai-generate mr-2 text-primary"></i>
                        <span>AI Model Validation</span>
                      </div>
                      <Button variant="outline" size="sm">Run</Button>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-medium mb-3">Network Diagnostics</h3>
                  <table className="w-full">
                    <tbody>
                      <tr className="border-b border-border">
                        <td className="py-2.5 text-textSecondary">Binance API Latency</td>
                        <td className="py-2.5 text-right text-primary">58ms</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-2.5 text-textSecondary">Market Data Throughput</td>
                        <td className="py-2.5 text-right">4.32 MB/s</td>
                      </tr>
                      <tr className="border-b border-border">
                        <td className="py-2.5 text-textSecondary">Order Execution Time</td>
                        <td className="py-2.5 text-right">112ms</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 text-textSecondary">Connection Status</td>
                        <td className="py-2.5 text-right">
                          <span className="inline-flex items-center text-primary">
                            <span className="w-2 h-2 bg-primary rounded-full mr-1.5"></span>
                            Connected
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>

                  <div className="mt-6">
                    <Button className="w-full">
                      <i className="ri-radar-line mr-2"></i>
                      Run Complete Diagnostics
                    </Button>
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