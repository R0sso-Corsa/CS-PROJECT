CREATE DATABASE IF NOT EXISTS cs_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cs_project;

CREATE TABLE IF NOT EXISTS app_users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tickers (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    slug VARCHAR(40) NOT NULL UNIQUE,
    display_name VARCHAR(120) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_jobs (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NULL,
    ticker_id INT UNSIGNED NOT NULL,
    requested_ticker VARCHAR(20) NOT NULL,
    status ENUM('queued', 'running', 'completed', 'failed') NOT NULL DEFAULT 'queued',
    requested_device VARCHAR(20) NOT NULL DEFAULT 'gpu',
    requested_epochs INT UNSIGNED NOT NULL DEFAULT 40,
    requested_batch_size INT UNSIGNED NOT NULL DEFAULT 16,
    requested_prediction_days INT UNSIGNED NOT NULL DEFAULT 30,
    requested_future_days INT UNSIGNED NOT NULL DEFAULT 30,
    requested_mc_runs INT UNSIGNED NOT NULL DEFAULT 100,
    output_message TEXT NULL,
    failure_message TEXT NULL,
    remote_manifest_json LONGTEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL DEFAULT NULL,
    completed_at TIMESTAMP NULL DEFAULT NULL,
    CONSTRAINT fk_prediction_jobs_user
        FOREIGN KEY (user_id) REFERENCES app_users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_prediction_jobs_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON DELETE CASCADE,
    INDEX idx_prediction_jobs_status_created (status, created_at),
    INDEX idx_prediction_jobs_ticker_created (ticker_id, created_at)
);

CREATE TABLE IF NOT EXISTS saved_graphs (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    job_id INT UNSIGNED NOT NULL UNIQUE,
    user_id INT UNSIGNED NULL,
    ticker_id INT UNSIGNED NOT NULL,
    title VARCHAR(180) NOT NULL,
    summary_text TEXT NULL,
    summary_plot_path VARCHAR(255) NULL,
    detail_plot_path VARCHAR(255) NULL,
    predictions_csv_path VARCHAR(255) NULL,
    forecast_csv_path VARCHAR(255) NULL,
    remote_job_directory VARCHAR(255) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_saved_graphs_job
        FOREIGN KEY (job_id) REFERENCES prediction_jobs(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_saved_graphs_user
        FOREIGN KEY (user_id) REFERENCES app_users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_saved_graphs_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON DELETE CASCADE,
    INDEX idx_saved_graphs_ticker_created (ticker_id, created_at)
);

CREATE TABLE IF NOT EXISTS saved_graph_assets (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    graph_id INT UNSIGNED NOT NULL,
    asset_kind ENUM('summary', 'detail', 'residuals') NOT NULL,
    mime_type VARCHAR(120) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    binary_data LONGBLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_saved_graph_assets_graph
        FOREIGN KEY (graph_id) REFERENCES saved_graphs(id)
        ON DELETE CASCADE,
    UNIQUE KEY uniq_graph_asset_kind (graph_id, asset_kind)
);

-- For a real account, create a user with PHP and `password_hash()`.
-- Until then, the site also accepts the demo fallback from `web/includes/config.php`.
