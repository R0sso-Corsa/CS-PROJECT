<?php
declare(strict_types=1);

require_once __DIR__ . '/layout.php';

function ensure_storage_directories(): void
{
    foreach ([WEB_STORAGE_ROOT, GRAPH_IMPORT_ROOT, LOG_ROOT] as $path) {
        if (!is_dir($path)) {
            mkdir($path, 0775, true);
        }
    }
}

function normalise_ticker(string $ticker): string
{
    $ticker = strtoupper(trim($ticker));
    $ticker = preg_replace('/[^A-Z0-9.\-=]/', '', $ticker) ?? '';
    return $ticker;
}

function ticker_slug(string $ticker): string
{
    $slug = strtolower($ticker);
    $slug = preg_replace('/[^a-z0-9]+/', '-', $slug) ?? 'ticker';
    return trim($slug, '-') ?: 'ticker';
}

function fetch_dashboard_stats(PDO $pdo): array
{
    return [
        'queued_jobs' => (int) $pdo->query("SELECT COUNT(*) FROM prediction_jobs WHERE status = 'queued'")->fetchColumn(),
        'running_jobs' => (int) $pdo->query("SELECT COUNT(*) FROM prediction_jobs WHERE status = 'running'")->fetchColumn(),
        'saved_graphs' => (int) $pdo->query("SELECT COUNT(*) FROM saved_graphs")->fetchColumn(),
        'tracked_tickers' => (int) $pdo->query("SELECT COUNT(*) FROM tickers")->fetchColumn(),
    ];
}

function fetch_recent_jobs(PDO $pdo, int $limit = 8): array
{
    $stmt = $pdo->prepare(
        "SELECT j.*, t.symbol AS ticker_symbol, u.username
         FROM prediction_jobs j
         INNER JOIN tickers t ON t.id = j.ticker_id
         LEFT JOIN app_users u ON u.id = j.user_id
         ORDER BY j.created_at DESC, j.id DESC
         LIMIT :limit"
    );
    $stmt->bindValue('limit', $limit, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function fetch_recent_graphs(PDO $pdo, int $limit = 8): array
{
    $stmt = $pdo->prepare(
        "SELECT g.*, t.symbol AS ticker_symbol, u.username
         FROM saved_graphs g
         INNER JOIN tickers t ON t.id = g.ticker_id
         LEFT JOIN app_users u ON u.id = g.user_id
         ORDER BY g.created_at DESC, g.id DESC
         LIMIT :limit"
    );
    $stmt->bindValue('limit', $limit, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function search_ticker_history(PDO $pdo, string $query): array
{
    $query = normalise_ticker($query);
    if ($query === '') {
        return [];
    }

    $stmt = $pdo->prepare(
        "SELECT
             t.*,
             COUNT(g.id) AS graph_count,
             MAX(g.created_at) AS latest_graph_at
         FROM tickers t
         LEFT JOIN saved_graphs g ON g.ticker_id = t.id
         WHERE t.symbol LIKE :query
         GROUP BY t.id
         ORDER BY t.symbol ASC"
    );
    $stmt->execute(['query' => '%' . $query . '%']);
    return $stmt->fetchAll();
}

function fetch_graphs_for_ticker(PDO $pdo, string $ticker, int $limit = 12): array
{
    $stmt = $pdo->prepare(
        "SELECT g.*, t.symbol AS ticker_symbol, u.username
         FROM saved_graphs g
         INNER JOIN tickers t ON t.id = g.ticker_id
         LEFT JOIN app_users u ON u.id = g.user_id
         WHERE t.symbol = :ticker
         ORDER BY g.created_at DESC, g.id DESC
         LIMIT :limit"
    );
    $stmt->bindValue('ticker', normalise_ticker($ticker));
    $stmt->bindValue('limit', $limit, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function fetch_jobs_for_ticker(PDO $pdo, string $ticker, int $limit = 8): array
{
    $stmt = $pdo->prepare(
        "SELECT j.*, t.symbol AS ticker_symbol, u.username
         FROM prediction_jobs j
         INNER JOIN tickers t ON t.id = j.ticker_id
         LEFT JOIN app_users u ON u.id = j.user_id
         WHERE t.symbol = :ticker
         ORDER BY j.created_at DESC, j.id DESC
         LIMIT :limit"
    );
    $stmt->bindValue('ticker', normalise_ticker($ticker));
    $stmt->bindValue('limit', $limit, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function get_or_create_ticker(PDO $pdo, string $ticker): array
{
    $ticker = normalise_ticker($ticker);
    $select = $pdo->prepare('SELECT * FROM tickers WHERE symbol = :symbol LIMIT 1');
    $select->execute(['symbol' => $ticker]);
    $row = $select->fetch();
    if (is_array($row)) {
        return $row;
    }

    $insert = $pdo->prepare(
        'INSERT INTO tickers (symbol, slug, display_name)
         VALUES (:symbol, :slug, :display_name)'
    );
    $insert->execute([
        'symbol' => $ticker,
        'slug' => ticker_slug($ticker),
        'display_name' => $ticker,
    ]);

    return [
        'id' => (int) $pdo->lastInsertId(),
        'symbol' => $ticker,
        'slug' => ticker_slug($ticker),
        'display_name' => $ticker,
    ];
}

function enqueue_prediction_job(PDO $pdo, ?int $userId, string $ticker): int
{
    $tickerRow = get_or_create_ticker($pdo, $ticker);

    $stmt = $pdo->prepare(
        "INSERT INTO prediction_jobs
            (user_id, ticker_id, requested_ticker, status, requested_device, requested_epochs, requested_batch_size, requested_prediction_days, requested_future_days, requested_mc_runs)
         VALUES
            (:user_id, :ticker_id, :requested_ticker, 'queued', :requested_device, :requested_epochs, :requested_batch_size, :requested_prediction_days, :requested_future_days, :requested_mc_runs)"
    );
    $stmt->bindValue('user_id', $userId, $userId === null ? PDO::PARAM_NULL : PDO::PARAM_INT);
    $stmt->bindValue('ticker_id', (int) $tickerRow['id'], PDO::PARAM_INT);
    $stmt->bindValue('requested_ticker', (string) $tickerRow['symbol']);
    $stmt->bindValue('requested_device', REMOTE_DEFAULT_DEVICE);
    $stmt->bindValue('requested_epochs', REMOTE_DEFAULT_EPOCHS, PDO::PARAM_INT);
    $stmt->bindValue('requested_batch_size', REMOTE_DEFAULT_BATCH_SIZE, PDO::PARAM_INT);
    $stmt->bindValue('requested_prediction_days', REMOTE_DEFAULT_PREDICTION_DAYS, PDO::PARAM_INT);
    $stmt->bindValue('requested_future_days', REMOTE_DEFAULT_FUTURE_DAYS, PDO::PARAM_INT);
    $stmt->bindValue('requested_mc_runs', REMOTE_DEFAULT_MC_RUNS, PDO::PARAM_INT);
    $stmt->execute();

    return (int) $pdo->lastInsertId();
}

function find_job(PDO $pdo, int $jobId): ?array
{
    $stmt = $pdo->prepare(
        "SELECT j.*, t.symbol AS ticker_symbol, g.id AS saved_graph_id, u.username
         FROM prediction_jobs j
         INNER JOIN tickers t ON t.id = j.ticker_id
         LEFT JOIN saved_graphs g ON g.job_id = j.id
         LEFT JOIN app_users u ON u.id = j.user_id
         WHERE j.id = :job_id
         LIMIT 1"
    );
    $stmt->execute(['job_id' => $jobId]);
    $row = $stmt->fetch();
    return is_array($row) ? $row : null;
}

function find_graph(PDO $pdo, int $graphId): ?array
{
    $stmt = $pdo->prepare(
        "SELECT g.*, t.symbol AS ticker_symbol, u.username
         FROM saved_graphs g
         INNER JOIN tickers t ON t.id = g.ticker_id
         LEFT JOIN app_users u ON u.id = g.user_id
         WHERE g.id = :graph_id
         LIMIT 1"
    );
    $stmt->execute(['graph_id' => $graphId]);
    $row = $stmt->fetch();
    return is_array($row) ? $row : null;
}

function graph_asset_exists(PDO $pdo, int $graphId, string $kind): bool
{
    $stmt = $pdo->prepare(
        "SELECT 1
         FROM saved_graph_assets
         WHERE graph_id = :graph_id AND asset_kind = :asset_kind
         LIMIT 1"
    );
    $stmt->execute([
        'graph_id' => $graphId,
        'asset_kind' => $kind,
    ]);
    return (bool) $stmt->fetchColumn();
}

function queue_position(PDO $pdo, int $jobId): ?int
{
    $job = find_job($pdo, $jobId);
    if ($job === null || $job['status'] !== 'queued') {
        return null;
    }

    $stmt = $pdo->prepare(
        "SELECT COUNT(*)
         FROM prediction_jobs
         WHERE status = 'queued'
           AND (created_at < :created_at OR (created_at = :created_at AND id <= :job_id))"
    );
    $stmt->execute([
        'created_at' => $job['created_at'],
        'job_id' => $job['id'],
    ]);
    return (int) $stmt->fetchColumn();
}

function spawn_queue_worker(): void
{
    ensure_storage_directories();
    $command = sprintf(
        'nohup php %s >> %s 2>&1 &',
        escapeshellarg(QUEUE_WORKER_SCRIPT),
        escapeshellarg(QUEUE_WORKER_LOG)
    );
    @exec($command);
}

function process_prediction_queue(PDO $pdo): void
{
    ensure_storage_directories();

    $lockHandle = fopen(QUEUE_WORKER_LOCK, 'c+');
    if ($lockHandle === false) {
        throw new RuntimeException('Unable to create the queue worker lock file.');
    }

    if (!flock($lockHandle, LOCK_EX | LOCK_NB)) {
        fclose($lockHandle);
        return;
    }

    try {
        while (true) {
            $job = claim_next_job($pdo);
            if ($job === null) {
                break;
            }

            try {
                $manifest = run_remote_training_job($job);
                $imports = import_remote_manifest($manifest, $job);
                $graphId = save_graph_from_job($pdo, $job, $manifest, $imports);
                mark_job_completed($pdo, (int) $job['id'], $graphId, $manifest);
            } catch (Throwable $error) {
                mark_job_failed($pdo, (int) $job['id'], $error->getMessage());
            }
        }
    } finally {
        flock($lockHandle, LOCK_UN);
        fclose($lockHandle);
    }
}

function claim_next_job(PDO $pdo): ?array
{
    $pdo->beginTransaction();

    try {
        $running = (int) $pdo->query("SELECT COUNT(*) FROM prediction_jobs WHERE status = 'running'")->fetchColumn();
        if ($running > 0) {
            $pdo->commit();
            return null;
        }

        $stmt = $pdo->query(
            "SELECT id
             FROM prediction_jobs
             WHERE status = 'queued'
             ORDER BY created_at ASC, id ASC
             LIMIT 1
             FOR UPDATE"
        );
        $row = $stmt->fetch();

        if (!is_array($row)) {
            $pdo->commit();
            return null;
        }

        $update = $pdo->prepare(
            "UPDATE prediction_jobs
             SET status = 'running',
                 started_at = NOW(),
                 failure_message = NULL
             WHERE id = :job_id"
        );
        $update->execute(['job_id' => (int) $row['id']]);
        $pdo->commit();
    } catch (Throwable $error) {
        $pdo->rollBack();
        throw $error;
    }

    return find_job($pdo, (int) $row['id']);
}

function mark_job_completed(PDO $pdo, int $jobId, int $graphId, array $manifest): void
{
    $stmt = $pdo->prepare(
        "UPDATE prediction_jobs
         SET status = 'completed',
             completed_at = NOW(),
             output_message = :output_message,
             remote_manifest_json = :remote_manifest_json
         WHERE id = :job_id"
    );
    $stmt->execute([
        'job_id' => $jobId,
        'output_message' => 'Training finished successfully and the graph was imported.',
        'remote_manifest_json' => json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES),
    ]);
}

function mark_job_failed(PDO $pdo, int $jobId, string $message): void
{
    $stmt = $pdo->prepare(
        "UPDATE prediction_jobs
         SET status = 'failed',
             completed_at = NOW(),
             failure_message = :failure_message
         WHERE id = :job_id"
    );
    $stmt->execute([
        'job_id' => $jobId,
        'failure_message' => $message,
    ]);
}

function build_remote_command(array $job): string
{
    $remoteScript = str_replace('\\', '/', REMOTE_REPO_ROOT) . '/web/tools/run_remote_prediction_job.py';

    $parts = [
        escapeshellarg(REMOTE_PYTHON),
        escapeshellarg($remoteScript),
        '--ticker',
        escapeshellarg((string) $job['requested_ticker']),
        '--job-id',
        escapeshellarg((string) $job['id']),
        '--device',
        escapeshellarg((string) $job['requested_device']),
        '--epochs',
        escapeshellarg((string) $job['requested_epochs']),
        '--batch-size',
        escapeshellarg((string) $job['requested_batch_size']),
        '--prediction-days',
        escapeshellarg((string) $job['requested_prediction_days']),
        '--future-days',
        escapeshellarg((string) $job['requested_future_days']),
        '--mc-runs',
        escapeshellarg((string) $job['requested_mc_runs']),
        '--output-root',
        escapeshellarg(REMOTE_OUTPUT_ROOT),
    ];

    return implode(' ', $parts);
}

function run_remote_training_job(array $job): array
{
    $sshTarget = REMOTE_SSH_USER . '@' . REMOTE_SSH_HOST;
    $remoteCommand = build_remote_command($job);
    $command = sprintf(
        'ssh -i %s -p %d %s %s',
        escapeshellarg(REMOTE_SSH_KEY),
        REMOTE_SSH_PORT,
        escapeshellarg($sshTarget),
        escapeshellarg($remoteCommand)
    );

    $output = [];
    $exitCode = 0;
    exec($command . ' 2>&1', $output, $exitCode);
    $raw = trim(implode("\n", $output));

    if ($exitCode !== 0) {
        throw new RuntimeException('Remote training failed: ' . $raw);
    }

    $manifest = json_decode($raw, true);
    if (is_array($manifest)) {
        return $manifest;
    }

    if (preg_match('/(\{.*\})/s', $raw, $matches) === 1) {
        $manifest = json_decode($matches[1], true);
        if (is_array($manifest)) {
            return $manifest;
        }
    }

    throw new RuntimeException('Remote training returned unreadable JSON manifest.');
}

function normalise_remote_scp_path(string $path): string
{
    $path = str_replace('\\', '/', $path);
    if (preg_match('/^[A-Za-z]:\//', $path) === 1) {
        return '/' . strtoupper($path[0]) . substr($path, 1);
    }
    return $path;
}

function scp_remote_file(string $remotePath, string $localPath): void
{
    $scpSource = REMOTE_SSH_USER . '@' . REMOTE_SSH_HOST . ':' . normalise_remote_scp_path($remotePath);
    $command = sprintf(
        'scp -i %s -P %d %s %s',
        escapeshellarg(REMOTE_SSH_KEY),
        REMOTE_SSH_PORT,
        escapeshellarg($scpSource),
        escapeshellarg($localPath)
    );

    $output = [];
    $exitCode = 0;
    exec($command . ' 2>&1', $output, $exitCode);
    if ($exitCode !== 0) {
        throw new RuntimeException('SCP import failed: ' . trim(implode("\n", $output)));
    }
}

function import_remote_manifest(array $manifest, array $job): array
{
    $ticker = normalise_ticker((string) $job['requested_ticker']);
    $jobDir = GRAPH_IMPORT_ROOT . '/' . ticker_slug($ticker) . '/job_' . (int) $job['id'];
    if (!is_dir($jobDir)) {
        mkdir($jobDir, 0775, true);
    }

    $local = [
        'job_dir' => $jobDir,
        'summary_plot' => $jobDir . '/summary.png',
        'detail_plot' => $jobDir . '/detail.png',
        'predictions_csv' => $jobDir . '/predictions.csv',
        'forecast_csv' => $jobDir . '/forecast.csv',
        'training_log' => $jobDir . '/training.log',
        'manifest_json' => $jobDir . '/manifest.json',
    ];

    $remoteFiles = $manifest['files'] ?? [];
    $required = [
        'summary_plot' => $local['summary_plot'],
        'detail_plot' => $local['detail_plot'],
        'predictions_csv' => $local['predictions_csv'],
        'forecast_csv' => $local['forecast_csv'],
    ];

    foreach ($required as $key => $destination) {
        if (empty($remoteFiles[$key])) {
            throw new RuntimeException('Manifest is missing required file entry: ' . $key);
        }
        scp_remote_file((string) $remoteFiles[$key], $destination);
    }

    if (!empty($remoteFiles['training_log'])) {
        scp_remote_file((string) $remoteFiles['training_log'], $local['training_log']);
    }

    file_put_contents(
        $local['manifest_json'],
        json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES)
    );

    return $local;
}

function save_graph_from_job(PDO $pdo, array $job, array $manifest, array $imports): int
{
    $summary = $manifest['summary'] ?? null;
    $stats = $manifest['stats'] ?? [];

    $stmt = $pdo->prepare(
        "INSERT INTO saved_graphs
            (job_id, user_id, ticker_id, title, summary_text, summary_plot_path, detail_plot_path, predictions_csv_path, forecast_csv_path, remote_job_directory, created_at)
         VALUES
            (:job_id, :user_id, :ticker_id, :title, :summary_text, :summary_plot_path, :detail_plot_path, :predictions_csv_path, :forecast_csv_path, :remote_job_directory, NOW())"
    );
    $stmt->bindValue('job_id', (int) $job['id'], PDO::PARAM_INT);
    $stmt->bindValue('user_id', $job['user_id'], $job['user_id'] === null ? PDO::PARAM_NULL : PDO::PARAM_INT);
    $stmt->bindValue('ticker_id', (int) $job['ticker_id'], PDO::PARAM_INT);
    $stmt->bindValue('title', (string) $job['requested_ticker'] . ' forecast run #' . (int) $job['id']);
    $stmt->bindValue('summary_text', (string) ($summary ?? 'Queued training result imported from the remote forecasting machine.'));
    $stmt->bindValue('summary_plot_path', $imports['summary_plot']);
    $stmt->bindValue('detail_plot_path', $imports['detail_plot']);
    $stmt->bindValue('predictions_csv_path', $imports['predictions_csv']);
    $stmt->bindValue('forecast_csv_path', $imports['forecast_csv']);
    $stmt->bindValue('remote_job_directory', (string) ($manifest['job_dir'] ?? ''));
    $stmt->execute();

    $graphId = (int) $pdo->lastInsertId();

    foreach (['summary' => $imports['summary_plot'], 'detail' => $imports['detail_plot']] as $kind => $path) {
        $asset = $pdo->prepare(
            "INSERT INTO saved_graph_assets
                (graph_id, asset_kind, mime_type, original_name, binary_data)
             VALUES
                (:graph_id, :asset_kind, :mime_type, :original_name, :binary_data)"
        );
        $asset->bindValue('graph_id', $graphId, PDO::PARAM_INT);
        $asset->bindValue('asset_kind', $kind);
        $asset->bindValue('mime_type', 'image/png');
        $asset->bindValue('original_name', basename($path));
        $asset->bindValue('binary_data', file_get_contents($path), PDO::PARAM_LOB);
        $asset->execute();
    }

    if (!empty($stats)) {
        $statsStmt = $pdo->prepare(
            "UPDATE saved_graphs
             SET summary_text = CONCAT(summary_text, '\n\nRMSE: ', :rmse, ' | MAE: ', :mae, ' | Forecast days: ', :future_days)
             WHERE id = :graph_id"
        );
        $statsStmt->execute([
            'rmse' => isset($stats['rmse']) ? (string) $stats['rmse'] : 'n/a',
            'mae' => isset($stats['mae']) ? (string) $stats['mae'] : 'n/a',
            'future_days' => isset($stats['future_days']) ? (string) $stats['future_days'] : 'n/a',
            'graph_id' => $graphId,
        ]);
    }

    return $graphId;
}

