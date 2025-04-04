import { Express, Request, Response } from "express";
import { createServer, Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { 
  insertPositionSchema, 
  insertTradeSchema,
  insertBotConfigSchema,
  insertBacktestResultSchema,
  insertSystemSettingsSchema,
  insertApiKeySchema
} from "@shared/schema";
import { BinanceAPI } from "./binance-api";
import { AIService } from "./ai-service";
import { TechnicalAnalysis } from "./technical-analysis";
import { BacktestService } from "./backtest-service";

// Initialize services
const binanceAPI = new BinanceAPI();
const aiService = new AIService();
const technicalAnalysis = new TechnicalAnalysis();
const backtestService = new BacktestService();

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);

  // Mock user ID for demo purposes
  const userId = 1;
  const botId = 1;

  // ACCOUNT ROUTES
  app.get("/api/account", async (req: Request, res: Response) => {
    try {
      const accountInfo = await binanceAPI.getAccountInfo();
      res.json(accountInfo);
    } catch (error) {
      res.status(500).json({ message: `Error fetching account info: ${error}` });
    }
  });

  app.get("/api/account/daily-profit", async (req: Request, res: Response) => {
    try {
      const dailyProfit = await binanceAPI.getDailyProfit();
      res.json(dailyProfit);
    } catch (error) {
      res.status(500).json({ message: `Error fetching daily profit: ${error}` });
    }
  });

  app.get("/api/account/performance", async (req: Request, res: Response) => {
    try {
      const performance = await binanceAPI.getTradingPerformance();
      res.json(performance);
    } catch (error) {
      res.status(500).json({ message: `Error fetching trading performance: ${error}` });
    }
  });

  app.get("/api/account/price", async (req: Request, res: Response) => {
    const { symbol } = req.query;
    
    if (!symbol || typeof symbol !== 'string') {
      return res.status(400).json({ message: "Symbol parameter is required" });
    }
    
    try {
      const price = await binanceAPI.getMarketPrice(symbol);
      res.json({ price });
    } catch (error) {
      res.status(500).json({ message: `Error fetching market price: ${error}` });
    }
  });

  app.get("/api/account/symbols", async (req: Request, res: Response) => {
    try {
      const symbols = await binanceAPI.getAvailableSymbols();
      res.json(symbols);
    } catch (error) {
      res.status(500).json({ message: `Error fetching available symbols: ${error}` });
    }
  });

  // POSITION ROUTES
  app.get("/api/positions", async (req: Request, res: Response) => {
    try {
      const positions = await storage.getPositions(botId);
      res.json(positions);
    } catch (error) {
      res.status(500).json({ message: `Error fetching positions: ${error}` });
    }
  });

  app.get("/api/positions/:id", async (req: Request, res: Response) => {
    const { id } = req.params;
    
    try {
      const position = await storage.getPosition(parseInt(id));
      if (!position) {
        return res.status(404).json({ message: "Position not found" });
      }
      res.json(position);
    } catch (error) {
      res.status(500).json({ message: `Error fetching position: ${error}` });
    }
  });

  app.post("/api/positions", async (req: Request, res: Response) => {
    try {
      const validatedData = insertPositionSchema.parse(req.body);
      const position = await storage.createPosition({
        ...validatedData,
        botId
      });
      res.status(201).json(position);
    } catch (error) {
      res.status(400).json({ message: `Invalid position data: ${error}` });
    }
  });

  app.patch("/api/positions/:id", async (req: Request, res: Response) => {
    const { id } = req.params;
    
    try {
      const updateSchema = z.object({
        takeProfit: z.number().optional(),
        stopLoss: z.number().optional(),
      });
      
      const validatedData = updateSchema.parse(req.body);
      const updatedPosition = await storage.updatePosition(parseInt(id), validatedData);
      
      if (!updatedPosition) {
        return res.status(404).json({ message: "Position not found" });
      }
      
      res.json(updatedPosition);
    } catch (error) {
      res.status(400).json({ message: `Invalid update data: ${error}` });
    }
  });

  app.post("/api/positions/:id/close", async (req: Request, res: Response) => {
    const { id } = req.params;
    
    try {
      const closedPosition = await storage.closePosition(parseInt(id));
      
      if (!closedPosition) {
        return res.status(404).json({ message: "Position not found" });
      }
      
      // Create a trade record from the closed position
      if (closedPosition.pnl !== undefined && closedPosition.pnlPercent !== undefined) {
        await storage.createTrade({
          botId,
          positionId: closedPosition.id,
          symbol: closedPosition.symbol,
          side: closedPosition.side,
          size: closedPosition.size,
          entryPrice: closedPosition.entryPrice,
          exitPrice: closedPosition.markPrice || closedPosition.entryPrice,
          leverage: closedPosition.leverage,
          pnl: closedPosition.pnl,
          pnlPercent: closedPosition.pnlPercent,
          strategy: closedPosition.strategy,
          openTime: closedPosition.openTime,
          closeTime: closedPosition.closeTime || new Date(),
          aiScore: closedPosition.aiScore,
          aiNotes: closedPosition.aiNotes
        });
      }
      
      res.json({ success: true, message: "Position closed successfully" });
    } catch (error) {
      res.status(500).json({ message: `Error closing position: ${error}` });
    }
  });

  // TRADES ROUTES
  app.get("/api/trades", async (req: Request, res: Response) => {
    const page = parseInt(req.query.page as string) || 1;
    const pageSize = parseInt(req.query.pageSize as string) || 10;
    const offset = (page - 1) * pageSize;
    
    try {
      const result = await storage.getTrades(botId, pageSize, offset);
      res.json(result);
    } catch (error) {
      res.status(500).json({ message: `Error fetching trades: ${error}` });
    }
  });

  // BOT ROUTES
  app.get("/api/bot/settings", async (req: Request, res: Response) => {
    try {
      const config = await storage.getBotConfigurationByUserId(userId);
      
      if (!config) {
        return res.status(404).json({ message: "Bot configuration not found" });
      }
      
      res.json(config);
    } catch (error) {
      res.status(500).json({ message: `Error fetching bot settings: ${error}` });
    }
  });

  app.post("/api/bot/settings", async (req: Request, res: Response) => {
    try {
      const validatedData = insertBotConfigSchema.parse(req.body);
      const existingConfig = await storage.getBotConfigurationByUserId(userId);
      
      let config;
      if (existingConfig) {
        config = await storage.updateBotConfiguration(existingConfig.id, validatedData);
      } else {
        config = await storage.createBotConfiguration({
          ...validatedData,
          userId
        });
      }
      
      res.json(config);
    } catch (error) {
      res.status(400).json({ message: `Invalid bot settings: ${error}` });
    }
  });

  app.get("/api/bot/status", async (req: Request, res: Response) => {
    try {
      const config = await storage.getBotConfigurationByUserId(userId);
      
      if (!config) {
        return res.status(404).json({ message: "Bot configuration not found" });
      }
      
      const activeStrategies = config.strategyConfig.strategies
        .filter((s: any) => s.isActive)
        .map((s: any) => s.id);
      
      const botStatus = {
        isRunning: config.isActive,
        activeStrategies,
        currentParameters: {
          minPositionSize: config.minPositionSize,
          maxPositionSize: config.maxPositionSize,
          dcaLevels: config.dcaLevels,
          profitTarget: config.profitTarget,
          dailyProfitTarget: config.dailyProfitTarget
        }
      };
      
      res.json(botStatus);
    } catch (error) {
      res.status(500).json({ message: `Error fetching bot status: ${error}` });
    }
  });

  app.post("/api/bot/toggle", async (req: Request, res: Response) => {
    const { isActive } = req.body;
    
    if (typeof isActive !== 'boolean') {
      return res.status(400).json({ message: "isActive must be a boolean" });
    }
    
    try {
      const config = await storage.getBotConfigurationByUserId(userId);
      
      if (!config) {
        return res.status(404).json({ message: "Bot configuration not found" });
      }
      
      const updated = await storage.updateBotConfiguration(config.id, { isActive });
      res.json({ success: true, isActive });
    } catch (error) {
      res.status(500).json({ message: `Error toggling bot status: ${error}` });
    }
  });

  // AI ROUTES
  app.get("/api/ai/stats", async (req: Request, res: Response) => {
    try {
      const stats = await aiService.getModelStats();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ message: `Error fetching AI stats: ${error}` });
    }
  });

  app.get("/api/ai/insights", async (req: Request, res: Response) => {
    try {
      const insights = await aiService.getInsights();
      res.json(insights);
    } catch (error) {
      res.status(500).json({ message: `Error fetching AI insights: ${error}` });
    }
  });

  app.get("/api/ai/market-analysis", async (req: Request, res: Response) => {
    const { symbol, timeframe } = req.query;
    
    if (!symbol || !timeframe || typeof symbol !== 'string' || typeof timeframe !== 'string') {
      return res.status(400).json({ message: "Symbol and timeframe parameters are required" });
    }
    
    try {
      const analysis = await aiService.getMarketAnalysis(symbol, timeframe);
      res.json(analysis);
    } catch (error) {
      res.status(500).json({ message: `Error fetching market analysis: ${error}` });
    }
  });

  app.post("/api/ai/train", async (req: Request, res: Response) => {
    const { datasets, epochs, batchSize } = req.body;
    
    if (!datasets || !epochs || !batchSize) {
      return res.status(400).json({ message: "datasets, epochs, and batchSize are required" });
    }
    
    try {
      await aiService.trainModel({ datasets, epochs, batchSize });
      res.json({ success: true, message: "Training started successfully" });
    } catch (error) {
      res.status(500).json({ message: `Error starting training: ${error}` });
    }
  });

  app.post("/api/ai/pause", async (req: Request, res: Response) => {
    try {
      await aiService.pauseTraining();
      res.json({ success: true, message: "Training paused successfully" });
    } catch (error) {
      res.status(500).json({ message: `Error pausing training: ${error}` });
    }
  });

  app.post("/api/ai/resume", async (req: Request, res: Response) => {
    try {
      await aiService.resumeTraining();
      res.json({ success: true, message: "Training resumed successfully" });
    } catch (error) {
      res.status(500).json({ message: `Error resuming training: ${error}` });
    }
  });

  app.get("/api/ai/training-progress", async (req: Request, res: Response) => {
    try {
      const progress = await aiService.getTrainingProgress();
      res.json(progress);
    } catch (error) {
      res.status(500).json({ message: `Error fetching training progress: ${error}` });
    }
  });

  // TECHNICAL INDICATORS ROUTES
  app.get("/api/data/indicators/rsi", async (req: Request, res: Response) => {
    const { symbol, timeframe, period } = req.query;
    
    if (!symbol || !timeframe || typeof symbol !== 'string' || typeof timeframe !== 'string') {
      return res.status(400).json({ message: "Symbol and timeframe parameters are required" });
    }
    
    try {
      const rsiData = await technicalAnalysis.getRSI({
        symbol,
        timeframe,
        period: period ? parseInt(period as string) : undefined
      });
      res.json(rsiData);
    } catch (error) {
      res.status(500).json({ message: `Error calculating RSI: ${error}` });
    }
  });

  app.get("/api/data/indicators/ema", async (req: Request, res: Response) => {
    const { symbol, timeframe, period } = req.query;
    
    if (!symbol || !timeframe || typeof symbol !== 'string' || typeof timeframe !== 'string') {
      return res.status(400).json({ message: "Symbol and timeframe parameters are required" });
    }
    
    try {
      const emaData = await technicalAnalysis.getEMA({
        symbol,
        timeframe,
        period: period ? parseInt(period as string) : undefined
      });
      res.json(emaData);
    } catch (error) {
      res.status(500).json({ message: `Error calculating EMA: ${error}` });
    }
  });

  // BACKTEST ROUTES
  app.get("/api/backtest/results", async (req: Request, res: Response) => {
    try {
      const results = await storage.getBacktestResults(userId);
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: `Error fetching backtest results: ${error}` });
    }
  });

  app.post("/api/backtest/run", async (req: Request, res: Response) => {
    const { symbol, timeframe, startDate, endDate, initialCapital, strategies, monteCarloSimulations } = req.body;
    
    if (!symbol || !timeframe || !startDate || !endDate || !initialCapital || !strategies) {
      return res.status(400).json({ message: "Missing required parameters" });
    }
    
    try {
      const result = await backtestService.runBacktest({
        symbol,
        timeframe,
        startDate,
        endDate,
        initialCapital,
        strategies,
        monteCarloSimulations: monteCarloSimulations || 0
      });
      
      const savedResult = await storage.createBacktestResult({
        userId,
        name: `${symbol} ${timeframe} ${new Date().toISOString().split('T')[0]}`,
        startDate: new Date(startDate),
        endDate: new Date(endDate),
        symbols: [symbol],
        timeframe,
        initialCapital,
        finalCapital: result.finalCapital,
        totalTrades: result.totalTrades,
        winningTrades: result.winningTrades,
        losingTrades: result.losingTrades,
        winRate: result.winRate,
        profitFactor: result.profitFactor,
        maxDrawdown: result.maxDrawdown,
        strategyConfig: { strategies },
        monteCarloPaths: monteCarloSimulations || undefined,
        monteCarloResults: result.monteCarloResults || undefined
      });
      
      res.json(savedResult);
    } catch (error) {
      res.status(500).json({ message: `Error running backtest: ${error}` });
    }
  });

  app.delete("/api/backtest/results/:id", async (req: Request, res: Response) => {
    const { id } = req.params;
    
    try {
      const success = await storage.deleteBacktestResult(parseInt(id));
      
      if (!success) {
        return res.status(404).json({ message: "Backtest result not found" });
      }
      
      res.json({ success: true, message: "Backtest result deleted successfully" });
    } catch (error) {
      res.status(500).json({ message: `Error deleting backtest result: ${error}` });
    }
  });

  // DATA MANAGEMENT ROUTES
  app.get("/api/data/available", async (req: Request, res: Response) => {
    try {
      const data = await backtestService.getAvailableData();
      res.json(data);
    } catch (error) {
      res.status(500).json({ message: `Error fetching available data: ${error}` });
    }
  });

  app.post("/api/data/download", async (req: Request, res: Response) => {
    const { symbol, timeframe, startDate, endDate } = req.body;
    
    if (!symbol || !timeframe || !startDate || !endDate) {
      return res.status(400).json({ message: "Missing required parameters" });
    }
    
    try {
      await backtestService.downloadData({
        symbol,
        timeframe,
        startDate,
        endDate
      });
      
      res.json({ success: true, message: "Data download started" });
    } catch (error) {
      res.status(500).json({ message: `Error downloading data: ${error}` });
    }
  });

  // SYSTEM ROUTES
  app.get("/api/system/status", async (req: Request, res: Response) => {
    try {
      const status = {
        status: 'online',
        cpu: 28,
        ram: 42,
        gpu: 15,
        systemInfo: {
          cpu: 'i5 12400k',
          ram: '32GB',
          gpu: 'RX6600'
        }
      };
      
      res.json(status);
    } catch (error) {
      res.status(500).json({ message: `Error fetching system status: ${error}` });
    }
  });
  
  app.get("/api/system/connection-status", async (req: Request, res: Response) => {
    try {
      const isConnected = binanceAPI.apiKey && binanceAPI.apiSecret ? true : false;
      res.json({ isConnected });
    } catch (error) {
      res.status(500).json({ message: `Error checking connection status: ${error}` });
    }
  });

  app.get("/api/system/settings", async (req: Request, res: Response) => {
    try {
      const settings = await storage.getSystemSettings(userId);
      
      if (!settings) {
        return res.status(404).json({ message: "System settings not found" });
      }
      
      res.json(settings);
    } catch (error) {
      res.status(500).json({ message: `Error fetching system settings: ${error}` });
    }
  });

  app.post("/api/system/settings", async (req: Request, res: Response) => {
    try {
      const validatedData = insertSystemSettingsSchema.parse(req.body);
      const existingSettings = await storage.getSystemSettings(userId);
      
      let settings;
      if (existingSettings) {
        settings = await storage.updateSystemSettings(existingSettings.id, validatedData);
      } else {
        settings = await storage.createSystemSettings({
          ...validatedData,
          userId
        });
      }
      
      res.json(settings);
    } catch (error) {
      res.status(400).json({ message: `Invalid system settings: ${error}` });
    }
  });

  app.post("/api/system/api-keys", async (req: Request, res: Response) => {
    const { apiKey, apiSecret, isTestnet } = req.body;
    
    if (!apiKey || !apiSecret) {
      return res.status(400).json({ message: "API key and secret are required" });
    }
    
    try {
      const validatedData = insertApiKeySchema.parse({
        apiKey,
        apiSecret,
        isTestnet: isTestnet || true,
        userId,
        description: "User API Key",
        isActive: true
      });
      
      const existingKey = await storage.getApiKey(userId);
      
      let key;
      if (existingKey) {
        key = await storage.updateApiKey(existingKey.id, validatedData);
      } else {
        key = await storage.createApiKey(validatedData);
      }
      
      // Initialize and test Binance API with new keys
      try {
        await binanceAPI.setApiKeys(apiKey, apiSecret, isTestnet || true);
        res.json({ success: true, message: "API keys updated successfully" });
      } catch (apiError) {
        console.error('Binance API connection error:', apiError);
        res.status(400).json({ 
          message: `Could not connect to Binance API: ${apiError.message || 'Connection test failed'}` 
        });
      }
    } catch (error) {
      res.status(400).json({ message: `Invalid API key data: ${error}` });
    }
  });

  return httpServer;
}
