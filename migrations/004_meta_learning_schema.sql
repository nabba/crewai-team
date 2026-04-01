-- Meta-learning schema — tracks which modification strategies succeed.

-- Per (category, strategy) success statistics for UCB1 selection
CREATE TABLE IF NOT EXISTS modification.strategy_stats (
    id SERIAL PRIMARY KEY,
    feedback_category TEXT NOT NULL,
    modification_strategy TEXT NOT NULL,
    attempts INT DEFAULT 0,
    successes INT DEFAULT 0,
    total_improvement FLOAT DEFAULT 0.0,
    last_updated TIMESTAMPTZ DEFAULT now(),
    UNIQUE(feedback_category, modification_strategy)
);

CREATE INDEX IF NOT EXISTS idx_strategy_stats_cat
    ON modification.strategy_stats(feedback_category);
