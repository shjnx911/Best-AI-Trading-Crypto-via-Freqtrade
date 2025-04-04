import { pgTable, text, serial, integer, boolean, timestamp, jsonb, real, doublePrecision } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// API Key configuration
export const apiKeys = pgTable("api_keys", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
  apiKey: text("api_key").notNull(),
  apiSecret: text("api_secret").notNull(),
  description: text("description"),
  isTestnet: boolean("is_testnet").default(true),
  isActive: boolean("is_active").default(true),
});

// Bot configurations
export const botConfigurations = pgTable("bot_configurations", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
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
  botId: integer("bot_id").notNull(),
  symbol: text("symbol").notNull(),
  isActive: boolean("is_active").default(true),
});

// Open positions
export const positions = pgTable("positions", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").notNull(),
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
  botId: integer("bot_id").notNull(),
  positionId: integer("position_id").notNull(),
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

// Backtesting results
export const backtestResults = pgTable("backtest_results", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull(),
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
  userId: integer("user_id").notNull(),
  telegramEnabled: boolean("telegram_enabled").default(false),
  telegramToken: text("telegram_token"),
  telegramChatId: text("telegram_chat_id"),
  uiTheme: text("ui_theme").default("dark"),
  logLevel: text("log_level").default("info"),
  backupEnabled: boolean("backup_enabled").default(false),
  backupFrequency: text("backup_frequency").default("daily"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Define insert schemas
export const insertApiKeySchema = createInsertSchema(apiKeys).omit({ id: true });
export const insertBotConfigSchema = createInsertSchema(botConfigurations).omit({ id: true, createdAt: true, updatedAt: true });
export const insertTradingPairSchema = createInsertSchema(tradingPairs).omit({ id: true });
export const insertPositionSchema = createInsertSchema(positions).omit({ id: true, openTime: true, closeTime: true });
export const insertTradeSchema = createInsertSchema(trades).omit({ id: true });
export const insertBacktestResultSchema = createInsertSchema(backtestResults).omit({ id: true, createdAt: true });
export const insertSystemSettingsSchema = createInsertSchema(systemSettings).omit({ id: true, createdAt: true, updatedAt: true });

// Define types
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

export type BacktestResult = typeof backtestResults.$inferSelect;
export type InsertBacktestResult = z.infer<typeof insertBacktestResultSchema>;

export type SystemSetting = typeof systemSettings.$inferSelect;
export type InsertSystemSetting = z.infer<typeof insertSystemSettingsSchema>;
