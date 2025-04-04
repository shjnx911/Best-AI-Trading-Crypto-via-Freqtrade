CREATE TABLE "ai_models" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"name" text NOT NULL,
	"version" text NOT NULL,
	"type" text NOT NULL,
	"accuracy" real,
	"precision" real,
	"recall" real,
	"trained_at" timestamp DEFAULT now(),
	"architecture" text NOT NULL,
	"hyperparams" jsonb,
	"gpu_optimized" boolean DEFAULT false,
	"is_active" boolean DEFAULT true
);
--> statement-breakpoint
CREATE TABLE "ai_signals" (
	"id" serial PRIMARY KEY NOT NULL,
	"model_id" integer NOT NULL,
	"symbol" text NOT NULL,
	"timeframe" text NOT NULL,
	"direction" text NOT NULL,
	"confidence" real NOT NULL,
	"entry_price" real,
	"target_price" real,
	"stop_price" real,
	"created_at" timestamp DEFAULT now(),
	"expires_at" timestamp,
	"metadata" jsonb,
	"status" text DEFAULT 'active'
);
--> statement-breakpoint
CREATE TABLE "api_keys" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"api_key" text NOT NULL,
	"api_secret" text NOT NULL,
	"description" text,
	"is_testnet" boolean DEFAULT true,
	"is_active" boolean DEFAULT true
);
--> statement-breakpoint
CREATE TABLE "backtest_results" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"name" text NOT NULL,
	"start_date" timestamp NOT NULL,
	"end_date" timestamp NOT NULL,
	"symbols" text[] NOT NULL,
	"timeframe" text NOT NULL,
	"initial_capital" real NOT NULL,
	"final_capital" real NOT NULL,
	"total_trades" integer NOT NULL,
	"winning_trades" integer NOT NULL,
	"losing_trades" integer NOT NULL,
	"win_rate" real NOT NULL,
	"profit_factor" real NOT NULL,
	"max_drawdown" real NOT NULL,
	"strategy_config" jsonb NOT NULL,
	"monte_carlo_paths" integer,
	"monte_carlo_results" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "bot_configurations" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"name" text NOT NULL,
	"min_position_size" real NOT NULL,
	"max_position_size" real NOT NULL,
	"max_pairs" integer NOT NULL,
	"dca_levels" integer NOT NULL,
	"profit_target" real NOT NULL,
	"stop_loss" real NOT NULL,
	"daily_profit_target" real NOT NULL,
	"trailing_profit_enabled" boolean DEFAULT false,
	"trailing_profit_percent" real,
	"trailing_stop_loss_enabled" boolean DEFAULT false,
	"trailing_stop_loss_percent" real,
	"strategy_config" jsonb NOT NULL,
	"is_active" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "gpu_settings" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"gpu_enabled" boolean DEFAULT true,
	"gpu_type" text DEFAULT 'AMD',
	"gpu_model" text DEFAULT 'RX6600',
	"memory_allocation" integer DEFAULT 4000,
	"power_limit" integer DEFAULT 100,
	"cuda_enabled" boolean DEFAULT false,
	"rocm_enabled" boolean DEFAULT true,
	"optimization_level" text DEFAULT 'high',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "positions" (
	"id" serial PRIMARY KEY NOT NULL,
	"bot_id" integer NOT NULL,
	"symbol" text NOT NULL,
	"side" text NOT NULL,
	"size" real NOT NULL,
	"entry_price" real NOT NULL,
	"leverage" integer NOT NULL,
	"take_profit" real,
	"stop_loss" real,
	"strategy" text NOT NULL,
	"status" text NOT NULL,
	"pnl" real,
	"pnl_percent" real,
	"open_time" timestamp DEFAULT now(),
	"close_time" timestamp,
	"ai_score" integer,
	"ai_notes" text
);
--> statement-breakpoint
CREATE TABLE "system_settings" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"telegram_enabled" boolean DEFAULT false,
	"telegram_token" text,
	"telegram_chat_id" text,
	"ui_theme" text DEFAULT 'dark',
	"accent_color" text DEFAULT '#8E59FF',
	"font_size" text DEFAULT 'medium',
	"chart_style" text DEFAULT 'candles',
	"log_level" text DEFAULT 'info',
	"backup_enabled" boolean DEFAULT false,
	"backup_frequency" text DEFAULT 'daily',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "trades" (
	"id" serial PRIMARY KEY NOT NULL,
	"bot_id" integer NOT NULL,
	"position_id" integer NOT NULL,
	"symbol" text NOT NULL,
	"side" text NOT NULL,
	"size" real NOT NULL,
	"entry_price" real NOT NULL,
	"exit_price" real NOT NULL,
	"leverage" integer NOT NULL,
	"pnl" real NOT NULL,
	"pnl_percent" real NOT NULL,
	"strategy" text NOT NULL,
	"open_time" timestamp NOT NULL,
	"close_time" timestamp NOT NULL,
	"ai_score" integer,
	"ai_notes" text
);
--> statement-breakpoint
CREATE TABLE "trading_pairs" (
	"id" serial PRIMARY KEY NOT NULL,
	"bot_id" integer NOT NULL,
	"symbol" text NOT NULL,
	"is_active" boolean DEFAULT true
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"username" text NOT NULL,
	"email" text NOT NULL,
	"password_hash" text NOT NULL,
	"full_name" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	"last_login" timestamp,
	"is_active" boolean DEFAULT true,
	CONSTRAINT "users_username_unique" UNIQUE("username"),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
ALTER TABLE "ai_models" ADD CONSTRAINT "ai_models_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_signals" ADD CONSTRAINT "ai_signals_model_id_ai_models_id_fk" FOREIGN KEY ("model_id") REFERENCES "public"."ai_models"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "api_keys" ADD CONSTRAINT "api_keys_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "backtest_results" ADD CONSTRAINT "backtest_results_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "bot_configurations" ADD CONSTRAINT "bot_configurations_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "gpu_settings" ADD CONSTRAINT "gpu_settings_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "positions" ADD CONSTRAINT "positions_bot_id_bot_configurations_id_fk" FOREIGN KEY ("bot_id") REFERENCES "public"."bot_configurations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "system_settings" ADD CONSTRAINT "system_settings_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trades" ADD CONSTRAINT "trades_bot_id_bot_configurations_id_fk" FOREIGN KEY ("bot_id") REFERENCES "public"."bot_configurations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trades" ADD CONSTRAINT "trades_position_id_positions_id_fk" FOREIGN KEY ("position_id") REFERENCES "public"."positions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trading_pairs" ADD CONSTRAINT "trading_pairs_bot_id_bot_configurations_id_fk" FOREIGN KEY ("bot_id") REFERENCES "public"."bot_configurations"("id") ON DELETE no action ON UPDATE no action;