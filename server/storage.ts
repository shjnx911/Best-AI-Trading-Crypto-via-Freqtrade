import {
  ApiKey, InsertApiKey,
  BotConfiguration, InsertBotConfiguration,
  TradingPair, InsertTradingPair,
  Position, InsertPosition,
  Trade, InsertTrade,
  BacktestResult, InsertBacktestResult,
  SystemSetting, InsertSystemSetting,
  User, InsertUser,
  GpuSetting, InsertGpuSetting,
  AiModel, InsertAiModel,
  AiSignal, InsertAiSignal,
  users, apiKeys, botConfigurations, tradingPairs, positions, trades, 
  backtestResults, systemSettings, gpuSettings, aiModels, aiSignals
} from "@shared/schema";
import { db } from './db';
import { eq, and, desc, gte, lte, sql } from 'drizzle-orm';
import { hash as bcryptHash, compare as bcryptCompare } from 'bcrypt';

export interface IStorage {
  // Người dùng
  getUserById?(id: number): Promise<User | undefined>;
  getUserByUsername?(username: string): Promise<User | undefined>;
  getUserByEmail?(email: string): Promise<User | undefined>;
  createUser?(user: InsertUser & { password: string }): Promise<User>;
  updateUser?(id: number, user: Partial<User>): Promise<User | undefined>;
  
  // GPU Settings
  getGpuSettings?(userId: number): Promise<GpuSetting | undefined>;
  createGpuSettings?(settings: InsertGpuSetting): Promise<GpuSetting>;
  updateGpuSettings?(id: number, settings: Partial<GpuSetting>): Promise<GpuSetting | undefined>;
  
  // AI Models
  getAiModels?(userId: number): Promise<AiModel[]>;
  getAiModel?(id: number): Promise<AiModel | undefined>;
  createAiModel?(model: InsertAiModel): Promise<AiModel>;
  updateAiModel?(id: number, model: Partial<AiModel>): Promise<AiModel | undefined>;
  deleteAiModel?(id: number): Promise<boolean>;
  
  // AI Signals
  getAiSignals?(modelId: number, limit?: number): Promise<AiSignal[]>;
  getAiSignalsBySymbol?(symbol: string, timeframe: string, limit?: number): Promise<AiSignal[]>;
  createAiSignal?(signal: InsertAiSignal): Promise<AiSignal>;
  updateAiSignalStatus?(id: number, status: string): Promise<AiSignal | undefined>;
  // API Keys
  getApiKey(userId: number): Promise<ApiKey | undefined>;
  createApiKey(apiKey: InsertApiKey): Promise<ApiKey>;
  updateApiKey(id: number, apiKey: Partial<ApiKey>): Promise<ApiKey | undefined>;
  deleteApiKey(id: number): Promise<boolean>;
  
  // Bot Configurations
  getBotConfiguration(id: number): Promise<BotConfiguration | undefined>;
  getBotConfigurationByUserId(userId: number): Promise<BotConfiguration | undefined>;
  createBotConfiguration(config: InsertBotConfiguration): Promise<BotConfiguration>;
  updateBotConfiguration(id: number, config: Partial<BotConfiguration>): Promise<BotConfiguration | undefined>;
  deleteBotConfiguration(id: number): Promise<boolean>;
  
  // Trading Pairs
  getTradingPairs(botId: number): Promise<TradingPair[]>;
  createTradingPair(pair: InsertTradingPair): Promise<TradingPair>;
  updateTradingPair(id: number, pair: Partial<TradingPair>): Promise<TradingPair | undefined>;
  deleteTradingPair(id: number): Promise<boolean>;
  
  // Positions
  getPositions(botId: number): Promise<Position[]>;
  getPosition(id: number): Promise<Position | undefined>;
  createPosition(position: InsertPosition): Promise<Position>;
  updatePosition(id: number, position: Partial<Position>): Promise<Position | undefined>;
  closePosition(id: number): Promise<Position | undefined>;
  
  // Trades
  getTrades(botId: number, limit?: number, offset?: number): Promise<{ trades: Trade[], total: number }>;
  getTrade(id: number): Promise<Trade | undefined>;
  createTrade(trade: InsertTrade): Promise<Trade>;
  
  // Backtest Results
  getBacktestResults(userId: number): Promise<BacktestResult[]>;
  getBacktestResult(id: number): Promise<BacktestResult | undefined>;
  createBacktestResult(result: InsertBacktestResult): Promise<BacktestResult>;
  deleteBacktestResult(id: number): Promise<boolean>;
  
  // System Settings
  getSystemSettings(userId: number): Promise<SystemSetting | undefined>;
  createSystemSettings(settings: InsertSystemSetting): Promise<SystemSetting>;
  updateSystemSettings(id: number, settings: Partial<SystemSetting>): Promise<SystemSetting | undefined>;
}

export class MemStorage implements IStorage {
  private apiKeys: Map<number, ApiKey>;
  private botConfigurations: Map<number, BotConfiguration>;
  private tradingPairs: Map<number, TradingPair>;
  private positions: Map<number, Position>;
  private trades: Map<number, Trade>;
  private backtestResults: Map<number, BacktestResult>;
  private systemSettings: Map<number, SystemSetting>;
  
  private apiKeyId: number = 1;
  private botConfigId: number = 1;
  private tradingPairId: number = 1;
  private positionId: number = 1;
  private tradeId: number = 1;
  private backtestResultId: number = 1;
  private systemSettingId: number = 1;
  
  constructor() {
    this.apiKeys = new Map();
    this.botConfigurations = new Map();
    this.tradingPairs = new Map();
    this.positions = new Map();
    this.trades = new Map();
    this.backtestResults = new Map();
    this.systemSettings = new Map();
    
    // Initialize with default data
    this.seedDefaultData();
  }
  
  private seedDefaultData() {
    // Add a default API key
    this.createApiKey({
      userId: 1,
      apiKey: "defaultApiKey",
      apiSecret: "defaultApiSecret",
      description: "Default API Key",
      isTestnet: true,
      isActive: true
    });
    
    // Add a default bot configuration
    this.createBotConfiguration({
      userId: 1,
      name: "Default Bot",
      minPositionSize: 2,
      maxPositionSize: 5,
      maxPairs: 3,
      dcaLevels: 3,
      profitTarget: 2.5,
      stopLoss: 1.5,
      dailyProfitTarget: 2.4,
      trailingProfitEnabled: true,
      trailingProfitPercent: 0.5,
      trailingStopLossEnabled: true,
      trailingStopLossPercent: 0.5,
      strategyConfig: {
        strategies: [
          { id: "smc", isActive: true, params: {} },
          { id: "trend", isActive: true, params: {} },
          { id: "vwap", isActive: true, params: {} },
          { id: "breakout", isActive: false, params: {} }
        ]
      },
      isActive: true
    });
    
    // Add default trading pairs
    this.createTradingPair({
      botId: 1,
      symbol: "BTC/USDT",
      isActive: true
    });
    
    this.createTradingPair({
      botId: 1,
      symbol: "ETH/USDT",
      isActive: true
    });
    
    // Add some example positions
    this.createPosition({
      botId: 1,
      symbol: "BTC/USDT",
      side: "LONG",
      size: 0.12,
      entryPrice: 27245.80,
      leverage: 3,
      takeProfit: 28200.00,
      stopLoss: 26800.00,
      strategy: "Smart Money Concept",
      status: "OPEN",
      pnl: 48.80,
      pnlPercent: 1.48,
      aiScore: 92,
      aiNotes: "Strong bullish momentum identified"
    });
    
    this.createPosition({
      botId: 1,
      symbol: "ETH/USDT",
      side: "SHORT",
      size: 1.5,
      entryPrice: 1948.32,
      leverage: 3,
      takeProfit: 1900.00,
      stopLoss: 1980.00,
      strategy: "Trend Following",
      status: "OPEN",
      pnl: 39.21,
      pnlPercent: 1.34,
      aiScore: 87,
      aiNotes: "Bearish divergence detected"
    });
    
    // Add example trades
    this.createTrade({
      botId: 1,
      positionId: 1,
      symbol: "BTC/USDT",
      side: "LONG",
      size: 0.15,
      entryPrice: 26845.42,
      exitPrice: 27352.18,
      leverage: 3,
      pnl: 75.90,
      pnlPercent: 1.88,
      strategy: "Smart Money Concept",
      openTime: new Date("2023-05-18T09:45:22Z"),
      closeTime: new Date("2023-05-18T14:30:15Z"),
      aiScore: 92,
      aiNotes: "Well executed entry based on order block"
    });
    
    this.createTrade({
      botId: 1,
      positionId: 2,
      symbol: "ETH/USDT",
      side: "SHORT",
      size: 2.0,
      entryPrice: 1986.42,
      exitPrice: 1924.30,
      leverage: 3,
      pnl: 124.24,
      pnlPercent: 3.12,
      strategy: "Trend Following",
      openTime: new Date("2023-05-18T07:12:49Z"),
      closeTime: new Date("2023-05-18T10:45:32Z"),
      aiScore: 87,
      aiNotes: "Excellent trend identification"
    });
    
    // Add example backtest results
    this.createBacktestResult({
      userId: 1,
      name: "BTC 4H Backtest",
      startDate: new Date("2023-01-01"),
      endDate: new Date("2023-04-30"),
      symbols: ["BTC/USDT"],
      timeframe: "4h",
      initialCapital: 10000,
      finalCapital: 12450,
      totalTrades: 78,
      winningTrades: 52,
      losingTrades: 26,
      winRate: 66.67,
      profitFactor: 2.3,
      maxDrawdown: 8.5,
      strategyConfig: {
        strategies: [
          { id: "smc", isActive: true, params: {} },
          { id: "trend", isActive: true, params: {} }
        ]
      },
      monteCarloPaths: 100,
      monteCarloResults: {
        avgReturn: 24.5,
        minReturn: 12.1,
        maxReturn: 35.8,
        standardDeviation: 5.2
      }
    });
    
    // Add system settings
    this.createSystemSettings({
      userId: 1,
      telegramEnabled: false,
      telegramToken: "",
      telegramChatId: "",
      uiTheme: "dark",
      logLevel: "info",
      backupEnabled: false,
      backupFrequency: "daily"
    });
  }
  
  // API Keys
  async getApiKey(userId: number): Promise<ApiKey | undefined> {
    for (const apiKey of this.apiKeys.values()) {
      if (apiKey.userId === userId) {
        return apiKey;
      }
    }
    return undefined;
  }
  
  async createApiKey(apiKey: InsertApiKey): Promise<ApiKey> {
    const id = this.apiKeyId++;
    const newApiKey: ApiKey = { ...apiKey, id };
    this.apiKeys.set(id, newApiKey);
    return newApiKey;
  }
  
  async updateApiKey(id: number, apiKey: Partial<ApiKey>): Promise<ApiKey | undefined> {
    const existing = this.apiKeys.get(id);
    if (!existing) return undefined;
    
    const updated = { ...existing, ...apiKey };
    this.apiKeys.set(id, updated);
    return updated;
  }
  
  async deleteApiKey(id: number): Promise<boolean> {
    return this.apiKeys.delete(id);
  }
  
  // Bot Configurations
  async getBotConfiguration(id: number): Promise<BotConfiguration | undefined> {
    return this.botConfigurations.get(id);
  }
  
  async getBotConfigurationByUserId(userId: number): Promise<BotConfiguration | undefined> {
    for (const config of this.botConfigurations.values()) {
      if (config.userId === userId) {
        return config;
      }
    }
    return undefined;
  }
  
  async createBotConfiguration(config: InsertBotConfiguration): Promise<BotConfiguration> {
    const id = this.botConfigId++;
    const now = new Date();
    const newConfig: BotConfiguration = { 
      ...config, 
      id, 
      createdAt: now, 
      updatedAt: now 
    };
    this.botConfigurations.set(id, newConfig);
    return newConfig;
  }
  
  async updateBotConfiguration(id: number, config: Partial<BotConfiguration>): Promise<BotConfiguration | undefined> {
    const existing = this.botConfigurations.get(id);
    if (!existing) return undefined;
    
    const updated = { 
      ...existing, 
      ...config, 
      updatedAt: new Date() 
    };
    this.botConfigurations.set(id, updated);
    return updated;
  }
  
  async deleteBotConfiguration(id: number): Promise<boolean> {
    return this.botConfigurations.delete(id);
  }
  
  // Trading Pairs
  async getTradingPairs(botId: number): Promise<TradingPair[]> {
    const pairs: TradingPair[] = [];
    for (const pair of this.tradingPairs.values()) {
      if (pair.botId === botId) {
        pairs.push(pair);
      }
    }
    return pairs;
  }
  
  async createTradingPair(pair: InsertTradingPair): Promise<TradingPair> {
    const id = this.tradingPairId++;
    const newPair: TradingPair = { ...pair, id };
    this.tradingPairs.set(id, newPair);
    return newPair;
  }
  
  async updateTradingPair(id: number, pair: Partial<TradingPair>): Promise<TradingPair | undefined> {
    const existing = this.tradingPairs.get(id);
    if (!existing) return undefined;
    
    const updated = { ...existing, ...pair };
    this.tradingPairs.set(id, updated);
    return updated;
  }
  
  async deleteTradingPair(id: number): Promise<boolean> {
    return this.tradingPairs.delete(id);
  }
  
  // Positions
  async getPositions(botId: number): Promise<Position[]> {
    const positions: Position[] = [];
    for (const position of this.positions.values()) {
      if (position.botId === botId && position.status === "OPEN") {
        positions.push(position);
      }
    }
    return positions;
  }
  
  async getPosition(id: number): Promise<Position | undefined> {
    return this.positions.get(id);
  }
  
  async createPosition(position: InsertPosition): Promise<Position> {
    const id = this.positionId++;
    const now = new Date();
    const newPosition: Position = { 
      ...position, 
      id, 
      openTime: now, 
      closeTime: undefined
    };
    this.positions.set(id, newPosition);
    return newPosition;
  }
  
  async updatePosition(id: number, position: Partial<Position>): Promise<Position | undefined> {
    const existing = this.positions.get(id);
    if (!existing) return undefined;
    
    const updated = { ...existing, ...position };
    this.positions.set(id, updated);
    return updated;
  }
  
  async closePosition(id: number): Promise<Position | undefined> {
    const position = this.positions.get(id);
    if (!position) return undefined;
    
    const closed = { 
      ...position, 
      status: "CLOSED", 
      closeTime: new Date() 
    };
    this.positions.set(id, closed);
    return closed;
  }
  
  // Trades
  async getTrades(botId: number, limit: number = 10, offset: number = 0): Promise<{ trades: Trade[], total: number }> {
    const trades: Trade[] = [];
    for (const trade of this.trades.values()) {
      if (trade.botId === botId) {
        trades.push(trade);
      }
    }
    
    // Sort by closeTime descending
    trades.sort((a, b) => b.closeTime.getTime() - a.closeTime.getTime());
    
    const total = trades.length;
    const paginatedTrades = trades.slice(offset, offset + limit);
    
    return { trades: paginatedTrades, total };
  }
  
  async getTrade(id: number): Promise<Trade | undefined> {
    return this.trades.get(id);
  }
  
  async createTrade(trade: InsertTrade): Promise<Trade> {
    const id = this.tradeId++;
    const newTrade: Trade = { ...trade, id };
    this.trades.set(id, newTrade);
    return newTrade;
  }
  
  // Backtest Results
  async getBacktestResults(userId: number): Promise<BacktestResult[]> {
    const results: BacktestResult[] = [];
    for (const result of this.backtestResults.values()) {
      if (result.userId === userId) {
        results.push(result);
      }
    }
    // Sort by creation date, newest first
    return results.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }
  
  async getBacktestResult(id: number): Promise<BacktestResult | undefined> {
    return this.backtestResults.get(id);
  }
  
  async createBacktestResult(result: InsertBacktestResult): Promise<BacktestResult> {
    const id = this.backtestResultId++;
    const now = new Date();
    const newResult: BacktestResult = { ...result, id, createdAt: now };
    this.backtestResults.set(id, newResult);
    return newResult;
  }
  
  async deleteBacktestResult(id: number): Promise<boolean> {
    return this.backtestResults.delete(id);
  }
  
  // System Settings
  async getSystemSettings(userId: number): Promise<SystemSetting | undefined> {
    for (const settings of this.systemSettings.values()) {
      if (settings.userId === userId) {
        return settings;
      }
    }
    return undefined;
  }
  
  async createSystemSettings(settings: InsertSystemSetting): Promise<SystemSetting> {
    const id = this.systemSettingId++;
    const now = new Date();
    const newSettings: SystemSetting = { 
      ...settings, 
      id, 
      createdAt: now, 
      updatedAt: now 
    };
    this.systemSettings.set(id, newSettings);
    return newSettings;
  }
  
  async updateSystemSettings(id: number, settings: Partial<SystemSetting>): Promise<SystemSetting | undefined> {
    const existing = this.systemSettings.get(id);
    if (!existing) return undefined;
    
    const updated = { 
      ...existing, 
      ...settings, 
      updatedAt: new Date() 
    };
    this.systemSettings.set(id, updated);
    return updated;
  }
}

export class DatabaseStorage implements IStorage {
  // User related methods
  async getUserById(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }
  
  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user;
  }
  
  async getUserByEmail(email: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email));
    return user;
  }
  
  async createUser(user: InsertUser & { password: string }): Promise<User> {
    const { password, ...userData } = user;
    const passwordHash = await bcryptHash(password, 10);
    const [createdUser] = await db.insert(users).values({
      ...userData,
      passwordHash,
    }).returning();
    return createdUser;
  }
  
  async updateUser(id: number, userData: Partial<User>): Promise<User | undefined> {
    const [updatedUser] = await db
      .update(users)
      .set({ ...userData, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning();
    return updatedUser;
  }
  
  // GPU Settings
  async getGpuSettings(userId: number): Promise<GpuSetting | undefined> {
    const [settings] = await db
      .select()
      .from(gpuSettings)
      .where(eq(gpuSettings.userId, userId));
    return settings;
  }
  
  async createGpuSettings(settings: InsertGpuSetting): Promise<GpuSetting> {
    const [createdSettings] = await db
      .insert(gpuSettings)
      .values(settings)
      .returning();
    return createdSettings;
  }
  
  async updateGpuSettings(id: number, settings: Partial<GpuSetting>): Promise<GpuSetting | undefined> {
    const [updatedSettings] = await db
      .update(gpuSettings)
      .set({ ...settings, updatedAt: new Date() })
      .where(eq(gpuSettings.id, id))
      .returning();
    return updatedSettings;
  }
  
  // AI Models
  async getAiModels(userId: number): Promise<AiModel[]> {
    return db
      .select()
      .from(aiModels)
      .where(eq(aiModels.userId, userId));
  }
  
  async getAiModel(id: number): Promise<AiModel | undefined> {
    const [model] = await db
      .select()
      .from(aiModels)
      .where(eq(aiModels.id, id));
    return model;
  }
  
  async createAiModel(model: InsertAiModel): Promise<AiModel> {
    const [createdModel] = await db
      .insert(aiModels)
      .values(model)
      .returning();
    return createdModel;
  }
  
  async updateAiModel(id: number, model: Partial<AiModel>): Promise<AiModel | undefined> {
    const [updatedModel] = await db
      .update(aiModels)
      .set(model)
      .where(eq(aiModels.id, id))
      .returning();
    return updatedModel;
  }
  
  async deleteAiModel(id: number): Promise<boolean> {
    const result = await db
      .delete(aiModels)
      .where(eq(aiModels.id, id));
    return result.count > 0;
  }
  
  // AI Signals
  async getAiSignals(modelId: number, limit: number = 10): Promise<AiSignal[]> {
    return db
      .select()
      .from(aiSignals)
      .where(eq(aiSignals.modelId, modelId))
      .orderBy(desc(aiSignals.createdAt))
      .limit(limit);
  }
  
  async getAiSignalsBySymbol(symbol: string, timeframe: string, limit: number = 10): Promise<AiSignal[]> {
    return db
      .select()
      .from(aiSignals)
      .where(and(
        eq(aiSignals.symbol, symbol),
        eq(aiSignals.timeframe, timeframe)
      ))
      .orderBy(desc(aiSignals.createdAt))
      .limit(limit);
  }
  
  async createAiSignal(signal: InsertAiSignal): Promise<AiSignal> {
    const [createdSignal] = await db
      .insert(aiSignals)
      .values(signal)
      .returning();
    return createdSignal;
  }
  
  async updateAiSignalStatus(id: number, status: string): Promise<AiSignal | undefined> {
    const [updatedSignal] = await db
      .update(aiSignals)
      .set({ status })
      .where(eq(aiSignals.id, id))
      .returning();
    return updatedSignal;
  }
  
  // API Keys
  async getApiKey(userId: number): Promise<ApiKey | undefined> {
    const [apiKey] = await db
      .select()
      .from(apiKeys)
      .where(eq(apiKeys.userId, userId));
    return apiKey;
  }
  
  async createApiKey(apiKey: InsertApiKey): Promise<ApiKey> {
    const [createdApiKey] = await db
      .insert(apiKeys)
      .values(apiKey)
      .returning();
    return createdApiKey;
  }
  
  async updateApiKey(id: number, apiKey: Partial<ApiKey>): Promise<ApiKey | undefined> {
    const [updatedApiKey] = await db
      .update(apiKeys)
      .set(apiKey)
      .where(eq(apiKeys.id, id))
      .returning();
    return updatedApiKey;
  }
  
  async deleteApiKey(id: number): Promise<boolean> {
    await db
      .delete(apiKeys)
      .where(eq(apiKeys.id, id));
    return true;
  }
  
  // Bot Configurations
  async getBotConfiguration(id: number): Promise<BotConfiguration | undefined> {
    const [config] = await db
      .select()
      .from(botConfigurations)
      .where(eq(botConfigurations.id, id));
    return config;
  }
  
  async getBotConfigurationByUserId(userId: number): Promise<BotConfiguration | undefined> {
    const [config] = await db
      .select()
      .from(botConfigurations)
      .where(eq(botConfigurations.userId, userId));
    return config;
  }
  
  async createBotConfiguration(config: InsertBotConfiguration): Promise<BotConfiguration> {
    const [createdConfig] = await db
      .insert(botConfigurations)
      .values(config)
      .returning();
    return createdConfig;
  }
  
  async updateBotConfiguration(id: number, config: Partial<BotConfiguration>): Promise<BotConfiguration | undefined> {
    const [updatedConfig] = await db
      .update(botConfigurations)
      .set({ ...config, updatedAt: new Date() })
      .where(eq(botConfigurations.id, id))
      .returning();
    return updatedConfig;
  }
  
  async deleteBotConfiguration(id: number): Promise<boolean> {
    await db
      .delete(botConfigurations)
      .where(eq(botConfigurations.id, id));
    return true;
  }
  
  // Trading Pairs
  async getTradingPairs(botId: number): Promise<TradingPair[]> {
    return db
      .select()
      .from(tradingPairs)
      .where(eq(tradingPairs.botId, botId));
  }
  
  async createTradingPair(pair: InsertTradingPair): Promise<TradingPair> {
    const [createdPair] = await db
      .insert(tradingPairs)
      .values(pair)
      .returning();
    return createdPair;
  }
  
  async updateTradingPair(id: number, pair: Partial<TradingPair>): Promise<TradingPair | undefined> {
    const [updatedPair] = await db
      .update(tradingPairs)
      .set(pair)
      .where(eq(tradingPairs.id, id))
      .returning();
    return updatedPair;
  }
  
  async deleteTradingPair(id: number): Promise<boolean> {
    await db
      .delete(tradingPairs)
      .where(eq(tradingPairs.id, id));
    return true;
  }
  
  // Positions
  async getPositions(botId: number): Promise<Position[]> {
    return db
      .select()
      .from(positions)
      .where(and(
        eq(positions.botId, botId),
        eq(positions.status, "OPEN")
      ));
  }
  
  async getPosition(id: number): Promise<Position | undefined> {
    const [position] = await db
      .select()
      .from(positions)
      .where(eq(positions.id, id));
    return position;
  }
  
  async createPosition(position: InsertPosition): Promise<Position> {
    const [createdPosition] = await db
      .insert(positions)
      .values({
        ...position,
        openTime: new Date(),
      })
      .returning();
    return createdPosition;
  }
  
  async updatePosition(id: number, position: Partial<Position>): Promise<Position | undefined> {
    const [updatedPosition] = await db
      .update(positions)
      .set(position)
      .where(eq(positions.id, id))
      .returning();
    return updatedPosition;
  }
  
  async closePosition(id: number): Promise<Position | undefined> {
    const [closedPosition] = await db
      .update(positions)
      .set({
        status: "CLOSED",
        closeTime: new Date(),
      })
      .where(eq(positions.id, id))
      .returning();
    return closedPosition;
  }
  
  // Trades
  async getTrades(botId: number, limit: number = 10, offset: number = 0): Promise<{ trades: Trade[], total: number }> {
    const tradesList = await db
      .select()
      .from(trades)
      .where(eq(trades.botId, botId))
      .orderBy(desc(trades.closeTime))
      .limit(limit)
      .offset(offset);
    
    const [countResult] = await db
      .select({
        count: sql`count(*)`.mapWith(Number)
      })
      .from(trades)
      .where(eq(trades.botId, botId));
    
    return { trades: tradesList, total: countResult.count };
  }
  
  async getTrade(id: number): Promise<Trade | undefined> {
    const [trade] = await db
      .select()
      .from(trades)
      .where(eq(trades.id, id));
    return trade;
  }
  
  async createTrade(trade: InsertTrade): Promise<Trade> {
    const [createdTrade] = await db
      .insert(trades)
      .values(trade)
      .returning();
    return createdTrade;
  }
  
  // Backtest Results
  async getBacktestResults(userId: number): Promise<BacktestResult[]> {
    return db
      .select()
      .from(backtestResults)
      .where(eq(backtestResults.userId, userId))
      .orderBy(desc(backtestResults.createdAt));
  }
  
  async getBacktestResult(id: number): Promise<BacktestResult | undefined> {
    const [result] = await db
      .select()
      .from(backtestResults)
      .where(eq(backtestResults.id, id));
    return result;
  }
  
  async createBacktestResult(result: InsertBacktestResult): Promise<BacktestResult> {
    const [createdResult] = await db
      .insert(backtestResults)
      .values(result)
      .returning();
    return createdResult;
  }
  
  async deleteBacktestResult(id: number): Promise<boolean> {
    await db
      .delete(backtestResults)
      .where(eq(backtestResults.id, id));
    return true;
  }
  
  // System Settings
  async getSystemSettings(userId: number): Promise<SystemSetting | undefined> {
    const [settings] = await db
      .select()
      .from(systemSettings)
      .where(eq(systemSettings.userId, userId));
    return settings;
  }
  
  async createSystemSettings(settings: InsertSystemSetting): Promise<SystemSetting> {
    const [createdSettings] = await db
      .insert(systemSettings)
      .values(settings)
      .returning();
    return createdSettings;
  }
  
  async updateSystemSettings(id: number, settings: Partial<SystemSetting>): Promise<SystemSetting | undefined> {
    const [updatedSettings] = await db
      .update(systemSettings)
      .set({ ...settings, updatedAt: new Date() })
      .where(eq(systemSettings.id, id))
      .returning();
    return updatedSettings;
  }
}

// Use database storage instead of in-memory storage
export const storage = new DatabaseStorage();
