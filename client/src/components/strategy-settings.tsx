import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { STRATEGIES, POSITION_SIZE_RANGE, DCA_LEVELS_RANGE, PROFIT_TARGET_RANGE, STOP_LOSS_RANGE, TRAILING_RANGE } from '@/lib/constants';

interface StrategySettingsProps {
  strategyConfig: any;
  onStrategyChange: (config: any) => void;
}

export function StrategySettings({ strategyConfig, onStrategyChange }: StrategySettingsProps) {
  const [selectedTab, setSelectedTab] = useState('general');

  const strategies = strategyConfig?.strategies || [];

  const updateGeneralSetting = (key: string, value: any) => {
    onStrategyChange({
      ...strategyConfig,
      [key]: value
    });
  };

  const toggleStrategy = (strategyId: string, isActive: boolean) => {
    const updatedStrategies = strategies.map((strategy: any) => 
      strategy.id === strategyId ? { ...strategy, isActive } : strategy
    );

    onStrategyChange({
      ...strategyConfig,
      strategies: updatedStrategies
    });
  };

  const updateStrategyParams = (strategyId: string, params: any) => {
    const updatedStrategies = strategies.map((strategy: any) => 
      strategy.id === strategyId ? { ...strategy, params: { ...strategy.params, ...params } } : strategy
    );

    onStrategyChange({
      ...strategyConfig,
      strategies: updatedStrategies
    });
  };

  return (
    <Card className="border-border bg-surface">
      <CardHeader>
        <CardTitle>Strategy Settings</CardTitle>
        <CardDescription>Configure your trading strategies and parameters</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={selectedTab} onValueChange={setSelectedTab}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="general">General Settings</TabsTrigger>
            <TabsTrigger value="strategies">Strategies</TabsTrigger>
            <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="general">
            <div className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Position Size Range ({strategyConfig?.minPositionSize || POSITION_SIZE_RANGE.min}% - {strategyConfig?.maxPositionSize || POSITION_SIZE_RANGE.max}%)</Label>
                </div>
                <Slider 
                  value={[
                    strategyConfig?.minPositionSize || POSITION_SIZE_RANGE.min, 
                    strategyConfig?.maxPositionSize || POSITION_SIZE_RANGE.max
                  ]} 
                  min={POSITION_SIZE_RANGE.min} 
                  max={POSITION_SIZE_RANGE.max} 
                  step={0.1}
                  onValueChange={(values) => {
                    updateGeneralSetting('minPositionSize', values[0]);
                    updateGeneralSetting('maxPositionSize', values[1]);
                  }}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>DCA Levels: {strategyConfig?.dcaLevels || DCA_LEVELS_RANGE.min}</Label>
                </div>
                <Slider 
                  value={[strategyConfig?.dcaLevels || DCA_LEVELS_RANGE.min]} 
                  min={DCA_LEVELS_RANGE.min} 
                  max={DCA_LEVELS_RANGE.max} 
                  step={1}
                  onValueChange={([value]) => updateGeneralSetting('dcaLevels', value)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Take Profit: {strategyConfig?.profitTarget || PROFIT_TARGET_RANGE.min}%</Label>
                </div>
                <Slider 
                  value={[strategyConfig?.profitTarget || PROFIT_TARGET_RANGE.min]} 
                  min={PROFIT_TARGET_RANGE.min} 
                  max={PROFIT_TARGET_RANGE.max} 
                  step={0.1}
                  onValueChange={([value]) => updateGeneralSetting('profitTarget', value)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Stop Loss: {strategyConfig?.stopLoss || STOP_LOSS_RANGE.min}%</Label>
                </div>
                <Slider 
                  value={[strategyConfig?.stopLoss || STOP_LOSS_RANGE.min]} 
                  min={STOP_LOSS_RANGE.min} 
                  max={STOP_LOSS_RANGE.max} 
                  step={0.1}
                  onValueChange={([value]) => updateGeneralSetting('stopLoss', value)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Daily Profit Target: {strategyConfig?.dailyProfitTarget || 2}%</Label>
                </div>
                <Slider 
                  value={[strategyConfig?.dailyProfitTarget || 2]} 
                  min={0.5} 
                  max={10} 
                  step={0.1}
                  onValueChange={([value]) => updateGeneralSetting('dailyProfitTarget', value)}
                />
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="trailing-profit">Trailing Profit</Label>
                  <Switch 
                    id="trailing-profit" 
                    checked={strategyConfig?.trailingProfitEnabled || false}
                    onCheckedChange={(checked) => updateGeneralSetting('trailingProfitEnabled', checked)}
                  />
                </div>
                
                {strategyConfig?.trailingProfitEnabled && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Trailing Profit %: {strategyConfig?.trailingProfitPercent || TRAILING_RANGE.min}%</Label>
                    </div>
                    <Slider 
                      value={[strategyConfig?.trailingProfitPercent || TRAILING_RANGE.min]} 
                      min={TRAILING_RANGE.min} 
                      max={TRAILING_RANGE.max} 
                      step={0.1}
                      onValueChange={([value]) => updateGeneralSetting('trailingProfitPercent', value)}
                    />
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="trailing-stop-loss">Trailing Stop Loss</Label>
                  <Switch 
                    id="trailing-stop-loss" 
                    checked={strategyConfig?.trailingStopLossEnabled || false}
                    onCheckedChange={(checked) => updateGeneralSetting('trailingStopLossEnabled', checked)}
                  />
                </div>
                
                {strategyConfig?.trailingStopLossEnabled && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Trailing Stop Loss %: {strategyConfig?.trailingStopLossPercent || TRAILING_RANGE.min}%</Label>
                    </div>
                    <Slider 
                      value={[strategyConfig?.trailingStopLossPercent || TRAILING_RANGE.min]} 
                      min={TRAILING_RANGE.min} 
                      max={TRAILING_RANGE.max} 
                      step={0.1}
                      onValueChange={([value]) => updateGeneralSetting('trailingStopLossPercent', value)}
                    />
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="strategies">
            <Accordion type="single" collapsible className="space-y-4">
              {STRATEGIES.map((strategy) => {
                const strategyData = strategies.find((s: any) => s.id === strategy.id) || { isActive: false, params: {} };
                
                return (
                  <AccordionItem value={strategy.id} key={strategy.id} className="border-border">
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center justify-between w-full pr-4">
                        <span>{strategy.name}</span>
                        <Switch 
                          checked={strategyData.isActive}
                          onCheckedChange={(checked) => toggleStrategy(strategy.id, checked)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="pt-2 pb-1 space-y-3">
                        <p className="text-sm text-textSecondary">{strategy.description}</p>
                        
                        <div className="grid grid-cols-2 gap-2 mt-3">
                          {strategy.indicators.map((indicator, idx) => (
                            <div key={idx} className="flex items-center text-sm">
                              <div className="mr-2 w-2 h-2 rounded-full bg-primary"></div>
                              {indicator}
                            </div>
                          ))}
                        </div>
                        
                        {/* Strategy-specific parameters could be added here */}
                        {strategy.id === 'smc' && (
                          <div className="space-y-3 mt-4">
                            <div className="space-y-1">
                              <Label htmlFor={`${strategy.id}-liquidity-threshold`}>Liquidity Detection Threshold</Label>
                              <Input 
                                id={`${strategy.id}-liquidity-threshold`}
                                type="number"
                                value={strategyData.params?.liquidityThreshold || 3}
                                onChange={(e) => updateStrategyParams(strategy.id, { liquidityThreshold: parseFloat(e.target.value) })}
                              />
                            </div>
                          </div>
                        )}
                        
                        {strategy.id === 'trend' && (
                          <div className="space-y-3 mt-4">
                            <div className="space-y-1">
                              <Label htmlFor={`${strategy.id}-ema-period`}>EMA Period</Label>
                              <Input 
                                id={`${strategy.id}-ema-period`}
                                type="number"
                                value={strategyData.params?.emaPeriod || 50}
                                onChange={(e) => updateStrategyParams(strategy.id, { emaPeriod: parseInt(e.target.value) })}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                );
              })}
            </Accordion>
          </TabsContent>

          <TabsContent value="advanced">
            <div className="space-y-6">
              <div>
                <Label className="block mb-2">AI Position Sizing</Label>
                <div className="flex items-center space-x-2 mb-4">
                  <Switch 
                    id="ai-position-sizing" 
                    checked={strategyConfig?.aiPositionSizingEnabled || false}
                    onCheckedChange={(checked) => updateGeneralSetting('aiPositionSizingEnabled', checked)}
                  />
                  <Label htmlFor="ai-position-sizing">Enable AI-assisted position sizing</Label>
                </div>
                <p className="text-sm text-textSecondary">When enabled, AI will dynamically adjust position sizes within your min-max range based on confidence signals and market conditions.</p>
              </div>

              <div>
                <Label className="block mb-2">Paper Trading Switch</Label>
                <div className="flex items-center space-x-2 mb-4">
                  <Switch 
                    id="paper-trading" 
                    checked={strategyConfig?.paperTradingEnabled || false}
                    onCheckedChange={(checked) => updateGeneralSetting('paperTradingEnabled', checked)}
                  />
                  <Label htmlFor="paper-trading">Switch to paper trading when daily target reached</Label>
                </div>
                <p className="text-sm text-textSecondary">When enabled, the bot will switch to paper trading mode once your daily profit target is reached to train the AI without risking more capital.</p>
              </div>

              <div>
                <Label className="block mb-2">Maximum Simultaneous Trades</Label>
                <Input 
                  type="number"
                  value={strategyConfig?.maxSimultaneousTrades || 3}
                  onChange={(e) => updateGeneralSetting('maxSimultaneousTrades', parseInt(e.target.value))}
                  min={1}
                  max={10}
                />
                <p className="text-sm text-textSecondary mt-2">Maximum number of positions that can be open at the same time.</p>
              </div>

              <div>
                <Label className="block mb-2">Volatility Adjustment</Label>
                <div className="flex items-center space-x-2 mb-4">
                  <Switch 
                    id="volatility-adjustment" 
                    checked={strategyConfig?.volatilityAdjustmentEnabled || false}
                    onCheckedChange={(checked) => updateGeneralSetting('volatilityAdjustmentEnabled', checked)}
                  />
                  <Label htmlFor="volatility-adjustment">Adjust position size based on market volatility</Label>
                </div>
                <p className="text-sm text-textSecondary">Automatically reduces position size during high volatility periods to manage risk.</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
