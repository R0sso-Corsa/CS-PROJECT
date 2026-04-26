USE cs_project;

ALTER TABLE prediction_jobs
    ADD COLUMN IF NOT EXISTS requested_device VARCHAR(20) NOT NULL DEFAULT 'gpu' AFTER status,
    ADD COLUMN IF NOT EXISTS requested_epochs INT UNSIGNED NOT NULL DEFAULT 40 AFTER requested_device,
    ADD COLUMN IF NOT EXISTS requested_batch_size INT UNSIGNED NOT NULL DEFAULT 16 AFTER requested_epochs,
    ADD COLUMN IF NOT EXISTS requested_prediction_days INT UNSIGNED NOT NULL DEFAULT 30 AFTER requested_batch_size,
    ADD COLUMN IF NOT EXISTS requested_future_days INT UNSIGNED NOT NULL DEFAULT 30 AFTER requested_prediction_days,
    ADD COLUMN IF NOT EXISTS requested_mc_runs INT UNSIGNED NOT NULL DEFAULT 100 AFTER requested_future_days,
    ADD COLUMN IF NOT EXISTS output_message TEXT NULL AFTER requested_mc_runs,
    ADD COLUMN IF NOT EXISTS failure_message TEXT NULL AFTER output_message,
    ADD COLUMN IF NOT EXISTS remote_manifest_json LONGTEXT NULL AFTER failure_message,
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMP NULL DEFAULT NULL AFTER created_at,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP NULL DEFAULT NULL AFTER started_at;
