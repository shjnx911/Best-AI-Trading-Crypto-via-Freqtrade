import { pgTable, text, serial, integer, boolean, timestamp, jsonb, real, doublePrecision, primaryKey, uuid, uniqueIndex } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Người dùng
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  email: text("email").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  fullName: text("full_name"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  lastLogin: timestamp("last_login"),
  isActive: boolean("is_active").default(true),
});

// Thiết lập GPU
export const gpuSettings = pgTable("gpu_settings", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  gpuEnabled: boolean("gpu_enabled").default(true),
  gpuType: text("gpu_type").default("AMD"),
  gpuModel: text("gpu_model").default("RX6600"),
  memoryAllocation: integer("memory_allocation").default(4000), // MB
  powerLimit: integer("power_limit").default(100), // %
  cudaEnabled: boolean("cuda_enabled").default(false),
  rocmEnabled: boolean("rocm_enabled").default(true),
  optimizationLevel: text("optimization_level").default("high"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// API Key configuration
export const apiKeys = pgTable("api_keys", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  apiKey: text("api_key").notNull(),
  apiSecret: text("api_secret").notNull(),
  description: text("description"),
  isTestnet: boolean("is_testnet").default(true),
  isActive: boolean("is_active").default(true),
});

// Bot configurations
export const botConfigurations = pgTable("bot_configurations", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  name: text("name").notNull(),
  minPositionSize: real("min_position_size").notNull(), // % of capital
  maxPositionSize: real("max_position_size").notNull(), // % of capital
  maxPairs: integer("max_pairs").notNull(),
  dcaLevels: integer("dca_levels").notNull(),
  profitTarget: real("profit_target").notNull(), // % profit per trade
  stopLoss: real("stop_loss").notNull(), // % stop loss per trade
  dailyProfitTarget: real("daily_profit_target").notNull(), // % of capital
  trailingProfitEnabled: boolean("trailing_profit_enabled").default(false),
  trailingProfitPercent: real("trailing_profit_percent"),
  trailingStopLossEnabled: boolean("trailing_stop_loss_enabled").default(false),
  trailingStopLossPercent: real("trailing_stop_loss_percent"),
  strategyConfig: jsonb("strategy_config").notNull(), // JSON with strategy-specific configurations
  isActive: boolean("is_active").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Trading pairs
export const tradingPairs = pgTable("trading_pairs", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").notNull().references(() => botConfigurations.id),
  symbol: text("symbol").notNull(),
  isActive: boolean("is_active").default(true),
});

// Open positions
export const positions = pgTable("positions", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").notNull().references(() => botConfigurations.id),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), // LONG or SHORT
  size: real("size").notNull(),
  entryPrice: real("entry_price").notNull(),
  leverage: integer("leverage").notNull(),
  takeProfit: real("take_profit"),
  stopLoss: real("stop_loss"),
  strategy: text("strategy").notNull(),
  status: text("status").notNull(), // OPEN, CLOSED
  pnl: real("pnl"),
  pnlPercent: real("pnl_percent"),
  openTime: timestamp("open_time").defaultNow(),
  closeTime: timestamp("close_time"),
  aiScore: integer("ai_score"), // 0-100 confidence score
  aiNotes: text("ai_notes"),
});

// Trading history
export const trades = pgTable("trades", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").notNull().references(() => botConfigurations.id),
  positionId: integer("position_id").notNull().references(() => positions.id),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(),
  size: real("size").notNull(),
  entryPrice: real("entry_price").notNull(),
  exitPrice: real("exit_price").notNull(),
  leverage: integer("leverage").notNull(),
  pnl: real("pnl").notNull(),
  pnlPercent: real("pnl_percent").notNull(),
  strategy: text("strategy").notNull(),
  openTime: timestamp("open_time").notNull(),
  closeTime: timestamp("close_time").notNull(),
  aiScore: integer("ai_score"), // 0-100 confidence score
  aiNotes: text("ai_notes"),
});

// AI Models
export const aiModels = pgTable("ai_models", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  name: text("name").notNull(),
  version: text("version").notNull(),
  type: text("type").notNull(), // trend, reversal, breakout, etc.
  accuracy: real("accuracy"),
  precision: real("precision"),
  recall: real("recall"),
  trainedAt: timestamp("trained_at").defaultNow(),
  architecture: text("architecture").notNull(),
  hyperparams: jsonb("hyperparams"),
  gpuOptimized: boolean("gpu_optimized").default(false),
  isActive: boolean("is_active").default(true),
});

// Tín hiệu AI
export const aiSignals = pgTable("ai_signals", {
  id: serial("id").primaryKey(),
  modelId: integer("model_id").notNull().references(() => aiModels.id),
  symbol: text("symbol").notNull(),
  timeframe: text("timeframe").notNull(),
  direction: text("direction").notNull(), // BUY, SELL, NEUTRAL
  confidence: real("confidence").notNull(),
  entryPrice: real("entry_price"),
  targetPrice: real("target_price"),
  stopPrice: real("stop_price"),
  createdAt: timestamp("created_at").defaultNow(),
  expiresAt: timestamp("expires_at"),
  metadata: jsonb("metadata"),
  status: text("status").default("active"), // active, expired, executed
});

// Backtesting results
export const backtestResults = pgTable("backtest_results", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  name: text("name").notNull(),
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date").notNull(),
  symbols: text("symbols").array().notNull(),
  timeframe: text("timeframe").notNull(),
  initialCapital: real("initial_capital").notNull(),
  finalCapital: real("final_capital").notNull(),
  totalTrades: integer("total_trades").notNull(),
  winningTrades: integer("winning_trades").notNull(),
  losingTrades: integer("losing_trades").notNull(),
  winRate: real("win_rate").notNull(),
  profitFactor: real("profit_factor").notNull(),
  maxDrawdown: real("max_drawdown").notNull(),
  strategyConfig: jsonb("strategy_config").notNull(),
  monteCarloPaths: integer("monte_carlo_paths"),
  monteCarloResults: jsonb("monte_carlo_results"),
  createdAt: timestamp("created_at").defaultNow(),
});

// System settings
export const systemSettings = pgTable("system_settings", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  telegramEnabled: boolean("telegram_enabled").default(false),
  telegramToken: text("telegram_token"),
  telegramChatId: text("telegram_chat_id"),
  uiTheme: text("ui_theme").default("dark"),
  accentColor: text("accent_color").default("#8E59FF"),
  fontSize: text("font_size").default("medium"),
  chartStyle: text("chart_style").default("candles"),
  logLevel: text("log_level").default("info"),
  backupEnabled: boolean("backup_enabled").default(false),
  backupFrequency: text("backup_frequency").default("daily"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Định nghĩa quan hệ giữa các bảng
export const usersRelations = relations(users, ({ many }) => ({
  apiKeys: many(apiKeys),
  botConfigurations: many(botConfigurations),
  gpuSettings: many(gpuSettings),
  aiModels: many(aiModels),
  backtestResults: many(backtestResults),
  systemSettings: many(systemSettings),
}));

export const botConfigurationsRelations = relations(botConfigurations, ({ one, many }) => ({
  user: one(users, {
    fields: [botConfigurations.userId],
    references: [users.id],
  }),
  tradingPairs: many(tradingPairs),
  positions: many(positions),
}));

export const tradingPairsRelations = relations(tradingPairs, ({ one }) => ({
  botConfiguration: one(botConfigurations, {
    fields: [tradingPairs.botId],
    references: [botConfigurations.id],
  }),
}));

export const positionsRelations = relations(positions, ({ one, many }) => ({
  botConfiguration: one(botConfigurations, {
    fields: [positions.botId],
    references: [botConfigurations.id],
  }),
  trades: many(trades),
}));

export const tradesRelations = relations(trades, ({ one }) => ({
  botConfiguration: one(botConfigurations, {
    fields: [trades.botId],
    references: [botConfigurations.id],
  }),
  position: one(positions, {
    fields: [trades.positionId],
    references: [positions.id],
  }),
}));

export const aiModelsRelations = relations(aiModels, ({ one, many }) => ({
  user: one(users, {
    fields: [aiModels.userId],
    references: [users.id],
  }),
  signals: many(aiSignals),
}));

export const aiSignalsRelations = relations(aiSignals, ({ one }) => ({
  model: one(aiModels, {
    fields: [aiSignals.modelId],
    references: [aiModels.id],
  }),
}));

// Define insert schemas
export const insertUserSchema = createInsertSchema(users).omit({ id: true, createdAt: true, updatedAt: true, lastLogin: true });
export const insertGpuSettingsSchema = createInsertSchema(gpuSettings).omit({ id: true, createdAt: true, updatedAt: true });
export const insertApiKeySchema = createInsertSchema(apiKeys).omit({ id: true });
export const insertBotConfigSchema = createInsertSchema(botConfigurations).omit({ id: true, createdAt: true, updatedAt: true });
export const insertTradingPairSchema = createInsertSchema(tradingPairs).omit({ id: true });
export const insertPositionSchema = createInsertSchema(positions).omit({ id: true, openTime: true, closeTime: true });
export const insertTradeSchema = createInsertSchema(trades).omit({ id: true });
export const insertAiModelSchema = createInsertSchema(aiModels).omit({ id: true, trainedAt: true });
export const insertAiSignalSchema = createInsertSchema(aiSignals).omit({ id: true, createdAt: true });
export const insertBacktestResultSchema = createInsertSchema(backtestResults).omit({ id: true, createdAt: true });
export const insertSystemSettingsSchema = createInsertSchema(systemSettings).omit({ id: true, createdAt: true, updatedAt: true });

// Define types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;

export type GpuSetting = typeof gpuSettings.$inferSelect;
export type InsertGpuSetting = z.infer<typeof insertGpuSettingsSchema>;

export type ApiKey = typeof apiKeys.$inferSelect;
export type InsertApiKey = z.infer<typeof insertApiKeySchema>;

export type BotConfiguration = typeof botConfigurations.$inferSelect;
export type InsertBotConfiguration = z.infer<typeof insertBotConfigSchema>;

export type TradingPair = typeof tradingPairs.$inferSelect;
export type InsertTradingPair = z.infer<typeof insertTradingPairSchema>;

export type Position = typeof positions.$inferSelect;
export type InsertPosition = z.infer<typeof insertPositionSchema>;

export type Trade = typeof trades.$inferSelect;
export type InsertTrade = z.infer<typeof insertTradeSchema>;

export type AiModel = typeof aiModels.$inferSelect;
export type InsertAiModel = z.infer<typeof insertAiModelSchema>;

export type AiSignal = typeof aiSignals.$inferSelect;
export type InsertAiSignal = z.infer<typeof insertAiSignalSchema>;

export type BacktestResult = typeof backtestResults.$inferSelect;
export type InsertBacktestResult = z.infer<typeof insertBacktestResultSchema>;

export type SystemSetting = typeof systemSettings.$inferSelect;
export type InsertSystemSetting = z.infer<typeof insertSystemSettingsSchema>;
