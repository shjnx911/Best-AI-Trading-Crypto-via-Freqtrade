import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '@/lib/constants';
import { fetchActivePositions, closePosition, updatePosition } from '@/lib/binance';
import { formatCurrency, formatNumber } from '@/lib/utils';
import { ActivePosition } from '@shared/types';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';

export function PositionsTable() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState<ActivePosition | null>(null);
  const [newTakeProfit, setNewTakeProfit] = useState<string>('');
  const [newStopLoss, setNewStopLoss] = useState<string>('');

  // Fetch active positions
  const { data: positions, isLoading } = useQuery<ActivePosition[]>({
    queryKey: [API_ENDPOINTS.POSITIONS],
    queryFn: fetchActivePositions,
  });

  // Close position mutation
  const closeMutation = useMutation({
    mutationFn: closePosition,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [API_ENDPOINTS.POSITIONS] });
      queryClient.invalidateQueries({ queryKey: [API_ENDPOINTS.ACCOUNT] });
      toast({
        title: "Position closed",
        description: "The position has been closed successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to close position: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Update position mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number, data: { takeProfit?: number, stopLoss?: number } }) => 
      updatePosition(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [API_ENDPOINTS.POSITIONS] });
      setIsEditDialogOpen(false);
      toast({
        title: "Position updated",
        description: "Take profit and stop loss levels updated successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to update position: ${error}`,
        variant: "destructive",
      });
    },
  });

  // Handle position close
  const handleClosePosition = (id: number) => {
    if (confirm("Are you sure you want to close this position?")) {
      closeMutation.mutate(id);
    }
  };

  // Handle edit dialog
  const openEditDialog = (position: ActivePosition) => {
    setSelectedPosition(position);
    setNewTakeProfit(position.takeProfit.toString());
    setNewStopLoss(position.stopLoss.toString());
    setIsEditDialogOpen(true);
  };

  // Handle save changes
  const handleSaveChanges = () => {
    if (!selectedPosition) return;
    
    updateMutation.mutate({
      id: selectedPosition.id as number,
      data: {
        takeProfit: parseFloat(newTakeProfit),
        stopLoss: parseFloat(newStopLoss),
      }
    });
  };

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
    <div className="mb-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Active Positions</h2>
        <div className="flex items-center">
          <button 
            className="bg-surfaceLight rounded-md px-3 py-1.5 text-sm hover:bg-surfaceLight/80 mr-2"
            onClick={() => queryClient.invalidateQueries({ queryKey: [API_ENDPOINTS.POSITIONS] })}
          >
            <i className="ri-refresh-line mr-1"></i>
            Refresh
          </button>
          <select className="bg-surfaceLight text-textPrimary border border-border rounded-md px-3 py-1.5 text-sm hover:bg-surfaceLight/80">
            <option>Sort by PnL</option>
            <option>Sort by Entry Time</option>
            <option>Sort by Size</option>
          </select>
        </div>
      </div>
      
      <div className="bg-surface rounded-lg border border-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-xs text-textSecondary">
                <th className="px-4 py-3 text-left font-medium border-b border-border">Symbol</th>
                <th className="px-4 py-3 text-left font-medium border-b border-border">Side</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Size</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Entry Price</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Mark Price</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Take Profit</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Stop Loss</th>
                <th className="px-4 py-3 text-right font-medium border-b border-border">Unrealized PnL</th>
                <th className="px-4 py-3 text-center font-medium border-b border-border">Actions</th>
              </tr>
            </thead>
            <tbody>
              {positions && positions.length > 0 ? (
                positions.map((position, index) => (
                  <tr key={index} className="border-b border-border hover:bg-surfaceLight/40 text-sm">
                    <td className="px-4 py-3 font-medium">{position.symbol}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        position.side === 'LONG' ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-secondary'
                      }`}>
                        {position.side}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {formatNumber(position.size)} {position.symbol.split('/')[0]}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.entryPrice)}</td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.markPrice)}</td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.takeProfit)}</td>
                    <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.stopLoss)}</td>
                    <td className={`px-4 py-3 text-right font-mono ${
                      position.unrealizedPnl > 0 ? 'text-primary' : 'text-secondary'
                    }`}>
                      {position.unrealizedPnl > 0 ? '+' : ''}
                      {formatCurrency(position.unrealizedPnl)} ({position.unrealizedPnlPercent > 0 ? '+' : ''}
                      {formatNumber(position.unrealizedPnlPercent)}%)
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button 
                        className="text-textSecondary hover:text-textPrimary px-1"
                        onClick={() => openEditDialog(position)}
                      >
                        <i className="ri-edit-line"></i>
                      </button>
                      <button 
                        className="text-secondary hover:text-secondary/80 px-1"
                        onClick={() => handleClosePosition(position.id as number)}
                        disabled={closeMutation.isPending}
                      >
                        <i className="ri-close-circle-line"></i>
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-textSecondary">
                    No active positions found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Edit Position Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Position</DialogTitle>
          </DialogHeader>
          
          {selectedPosition && (
            <div className="space-y-4 py-2">
              <div className="flex justify-between px-1">
                <span className="text-textSecondary">Symbol:</span>
                <span className="font-medium">{selectedPosition.symbol}</span>
              </div>
              
              <div className="flex justify-between px-1">
                <span className="text-textSecondary">Side:</span>
                <span className={selectedPosition.side === 'LONG' ? 'text-primary' : 'text-secondary'}>
                  {selectedPosition.side}
                </span>
              </div>
              
              <div className="flex justify-between px-1">
                <span className="text-textSecondary">Entry Price:</span>
                <span className="font-mono">{formatCurrency(selectedPosition.entryPrice)}</span>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="takeProfit">Take Profit</Label>
                <Input
                  id="takeProfit"
                  type="number"
                  step="0.01"
                  value={newTakeProfit}
                  onChange={(e) => setNewTakeProfit(e.target.value)}
                  className="font-mono"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="stopLoss">Stop Loss</Label>
                <Input
                  id="stopLoss"
                  type="number"
                  step="0.01"
                  value={newStopLoss}
                  onChange={(e) => setNewStopLoss(e.target.value)}
                  className="font-mono"
                />
              </div>
            </div>
          )}
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>Cancel</Button>
            <Button 
              onClick={handleSaveChanges} 
              disabled={updateMutation.isPending}
              className="bg-primary hover:bg-primary/90 text-white"
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
