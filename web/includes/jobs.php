<?php

require_once __DIR__ . '/layout.php';

function ensure_storage_directories()
{
    foreach ([WEB_STORAGE_ROOT, GRAPH_IMPORT_ROOT, LOG_ROOT] as $path) {
        if (!is_dir($path)) {
            mkdir($path, 0775, true);
        }
    }
}

function normalise_ticker($ticker)
{
    $ticker = strtoupper(trim($ticker));
    $ticker = preg_replace('/[^A-Z0-9.\-=]/', '', $ticker);
    if ($ticker === null) {
        $ticker = '';
    }
    return $ticker;
}

function normalise_search_query($query)
{
    $query = trim((string) $query);
    $query = preg_replace('/\s+/', ' ', $query);
    if ($query === null) {
        $query = '';
    }
    return $query;
}

function ticker_slug($ticker)
{
    $slug = strtolower($ticker);
    $slug = preg_replace('/[^a-z0-9]+/', '-', $slug);
    if ($slug === null) {
        $slug = 'ticker';
    }
    return trim($slug, '-') ?: 'ticker';
}

function unique_ticker_slug($pdo, $ticker)
{
    $base = ticker_slug($ticker);
    $slug = $base;
    $attempt = 1;

    while (true) {
        $stmt = $pdo->prepare('SELECT symbol FROM tickers WHERE slug = :slug LIMIT 1');
        $stmt->execute(['slug' => $slug]);
        $row = $stmt->fetch();

        if (!is_array($row) || (isset($row['symbol']) && (string) $row['symbol'] === normalise_ticker($ticker))) {
            return $slug;
        }

        $suffix = $attempt === 1 ? substr(sha1(normalise_ticker($ticker)), 0, 8) : (string) ($attempt + 1);
        $slug = substr($base, 0, max(1, 39 - strlen($suffix))) . '-' . $suffix;
        $attempt++;
    }
}

function fetch_dashboard_stats($pdo)
{
    return [
        'queued_jobs' => (int) $pdo->query("SELECT COUNT(*) FROM prediction_jobs WHERE status = 'queued'")->fetchColumn(),
        'running_jobs' => (int) $pdo->query("SELECT COUNT(*) FROM prediction_jobs WHERE status = 'running'")->fetchColumn(),
        'saved_graphs' => (int) $pdo->query("SELECT COUNT(*) FROM saved_graphs")->fetchColumn(),
        'tracked_tickers' => (int) $pdo->query("SELECT COUNT(*) FROM tickers")->fetchColumn(),
    ];
}

function fetch_recent_jobs($pdo, $limit = 8)
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

function fetch_recent_graphs($pdo, $limit = 8)
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

function search_ticker_history($pdo, $query)
{
    $query = normalise_search_query($query);
    if ($query === '') {
        return [];
    }

    $terms = preg_split('/\s+/', strtolower($query));
    if (!is_array($terms)) {
        $terms = [];
    }

    $terms = array_values(array_filter(array_unique($terms), static function ($term) {
        return $term !== '';
    }));

    if ($terms === []) {
        return [];
    }

    $whereParts = [];
    $params = [];

    foreach ($terms as $index => $term) {
        $textKey = 'text_' . $index;
        $symbolKey = 'symbol_' . $index;
        $slugKey = 'slug_' . $index;

        $params[$textKey] = '%' . $term . '%';
        $params[$symbolKey] = '%' . normalise_ticker($term) . '%';
        $params[$slugKey] = '%' . ticker_slug($term) . '%';

        $whereParts[] =
            "(t.symbol LIKE :$symbolKey
              OR t.slug LIKE :$slugKey
              OR t.display_name LIKE :$textKey
              OR g.title LIKE :$textKey
              OR g.summary_text LIKE :$textKey
              OR j.requested_ticker LIKE :$symbolKey)";
    }

    $sql =
        "SELECT
             t.*,
             COUNT(DISTINCT g.id) AS graph_count,
             MAX(g.created_at) AS latest_graph_at,
             COUNT(DISTINCT j.id) AS job_count,
             MAX(j.created_at) AS latest_job_at
         FROM tickers t
         LEFT JOIN saved_graphs g ON g.ticker_id = t.id
         LEFT JOIN prediction_jobs j ON j.ticker_id = t.id
         WHERE " . implode(' AND ', $whereParts) . "
         GROUP BY t.id
         ORDER BY
             CASE
                 WHEN t.symbol = :exact_symbol THEN 0
                 WHEN t.symbol LIKE :starts_symbol THEN 1
                 WHEN t.display_name LIKE :starts_text THEN 2
                 WHEN t.slug LIKE :starts_slug THEN 3
                 ELSE 4
             END,
             graph_count DESC,
             job_count DESC,
             t.symbol ASC
         LIMIT :limit";

    $stmt = $pdo->prepare($sql);

    foreach ($params as $key => $value) {
        $stmt->bindValue($key, $value);
    }

    $stmt->bindValue('exact_symbol', normalise_ticker($query));
    $stmt->bindValue('starts_symbol', normalise_ticker($query) . '%');
    $stmt->bindValue('starts_text', $query . '%');
    $stmt->bindValue('starts_slug', ticker_slug($query) . '%');
    $stmt->bindValue('limit', 25, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function search_yfinance_tickers($query, $limit = 10, &$error = null)
{
    ensure_storage_directories();

    $query = normalise_search_query($query);
    if ($query === '') {
        return [];
    }

    $script = yfinance_search_helper_path();
    if (!is_file($script)) {
        $error = 'The yfinance search helper is missing at ' . $script . '.';
        return [];
    }

    $command = build_yfinance_search_command($query, $limit);

    $output = [];
    $exitCode = 0;
    exec($command . ' 2>&1', $output, $exitCode);
    $raw = trim(implode("\n", $output));
    log_yfinance_search_debug($query, $command, $exitCode, $raw);

    if ($raw === '') {
        $error = 'The yfinance search helper returned no output.';
        return [];
    }

    $payload = json_decode($raw, true);
    if (!is_array($payload)) {
        foreach ($output as $line) {
            $candidate = trim((string) $line);
            if ($candidate === '' || $candidate[0] !== '{') {
                continue;
            }

            $decoded = json_decode($candidate, true);
            if (is_array($decoded)) {
                $payload = $decoded;
                break;
            }
        }
    }

    if (!is_array($payload)) {
        $error = 'The yfinance search helper returned unreadable output: ' . $raw;
        return [];
    }

    if (empty($payload['ok'])) {
        $error = isset($payload['error']) ? (string) $payload['error'] : 'The yfinance lookup failed.';
        return [];
    }

    if (!isset($payload['results']) || !is_array($payload['results'])) {
        log_yfinance_search_debug($query, $command, $exitCode, $raw, 0);
        return [];
    }

    $results = array_values(array_filter($payload['results'], static function ($row) {
        return is_array($row) && !empty($row['symbol']);
    }));
    log_yfinance_search_debug($query, $command, $exitCode, $raw, count($results));
    return $results;
}

function yfinance_search_helper_path()
{
    return WEB_ROOT . '/tools/search_yfinance.py';
}

function build_yfinance_search_command($query, $limit = 10)
{
    $prefix = defined('YFINANCE_SEARCH_ENV_PREFIX') ? trim((string) YFINANCE_SEARCH_ENV_PREFIX) : '';
    $command = sprintf(
        '%s %s %s --limit %d',
        escapeshellarg(YFINANCE_SEARCH_PYTHON),
        escapeshellarg(yfinance_search_helper_path()),
        escapeshellarg(normalise_search_query($query)),
        (int) $limit
    );

    return $prefix !== '' ? $prefix . ' ' . $command : $command;
}

function log_yfinance_search_marker($message, $context = [])
{
    if (!is_dir(LOG_ROOT)) {
        @mkdir(LOG_ROOT, 0775, true);
    }

    $line = '[' . date(DATE_ATOM) . '] ' . $message;
    if ($context !== []) {
        $line .= ' ' . json_encode($context, JSON_UNESCAPED_SLASHES);
    }
    $line .= "\n";

    $logPath = LOG_ROOT . '/yfinance-search.log';
    $written = @file_put_contents($logPath, $line, FILE_APPEND);
    if ($written === false) {
        error_log('Unable to write yfinance marker log to ' . $logPath);
    }
}

function log_yfinance_search_debug($query, $command, $exitCode, $raw, $resultCount = null)
{
    foreach ([WEB_STORAGE_ROOT, LOG_ROOT] as $path) {
        if (!is_dir($path)) {
            @mkdir($path, 0775, true);
        }
    }

    $message = '[' . date(DATE_ATOM) . '] yfinance search' . "\n";
    $message .= 'query=' . $query . "\n";
    $message .= 'command=' . $command . "\n";
    $message .= 'exit_code=' . (string) $exitCode . "\n";
    if ($resultCount !== null) {
        $message .= 'parsed_results=' . (string) $resultCount . "\n";
    }
    $message .= 'raw=' . ($raw !== '' ? $raw : '[empty]') . "\n\n";

    $logPath = LOG_ROOT . '/yfinance-search.log';
    $written = @file_put_contents($logPath, $message, FILE_APPEND);
    if ($written === false) {
        error_log('Unable to write yfinance search log to ' . $logPath);
    }
}

function fetch_graphs_for_ticker($pdo, $ticker, $limit = 12)
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

function fetch_jobs_for_ticker($pdo, $ticker, $limit = 8)
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

function get_or_create_ticker($pdo, $ticker)
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
        'slug' => unique_ticker_slug($pdo, $ticker),
        'display_name' => $ticker,
    ]);

    return [
        'id' => (int) $pdo->lastInsertId(),
        'symbol' => $ticker,
        'slug' => unique_ticker_slug($pdo, $ticker),
        'display_name' => $ticker,
    ];
}

function enqueue_prediction_job($pdo, $userId, $ticker)
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

function find_job($pdo, $jobId)
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

function find_graph($pdo, $graphId)
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

function graph_asset_exists($pdo, $graphId, $kind)
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

function queue_position($pdo, $jobId)
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

function spawn_queue_worker()
{
    ensure_storage_directories();
    $command = sprintf(
        'nohup %s %s >> %s 2>&1 &',
        escapeshellarg(PHP_CLI_BINARY),
        escapeshellarg(QUEUE_WORKER_SCRIPT),
        escapeshellarg(QUEUE_WORKER_LOG)
    );
    log_queue_worker('Spawning queue worker.', ['command' => $command]);
    @exec($command);
}

function process_prediction_queue($pdo)
{
    ensure_storage_directories();
    log_queue_worker('Queue worker started.');

    $lockHandle = fopen(QUEUE_WORKER_LOCK, 'c+');
    if ($lockHandle === false) {
        throw new RuntimeException('Unable to create the queue worker lock file.');
    }

    if (!flock($lockHandle, LOCK_EX | LOCK_NB)) {
        log_queue_worker('Another queue worker already holds the lock; exiting.');
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
                log_queue_worker('Claimed job.', ['job_id' => (int) $job['id'], 'ticker' => (string) $job['requested_ticker']]);
                $manifest = run_remote_training_job($job);
                $imports = import_remote_manifest($manifest, $job);
                $graphId = save_graph_from_job($pdo, $job, $manifest, $imports);
                mark_job_completed($pdo, (int) $job['id'], $graphId, $manifest);
                log_queue_worker('Completed job.', ['job_id' => (int) $job['id'], 'graph_id' => $graphId]);
            } catch (Throwable $error) {
                mark_job_failed($pdo, (int) $job['id'], $error->getMessage());
                log_queue_worker('Failed job.', ['job_id' => (int) $job['id'], 'error' => $error->getMessage()]);
            }
        }
    } finally {
        flock($lockHandle, LOCK_UN);
        fclose($lockHandle);
        log_queue_worker('Queue worker finished.');
    }
}

function log_queue_worker($message, $context = [])
{
    if (!is_dir(LOG_ROOT)) {
        @mkdir(LOG_ROOT, 0775, true);
    }

    $line = '[' . date(DATE_ATOM) . '] ' . $message;
    if ($context !== []) {
        $line .= ' ' . json_encode($context, JSON_UNESCAPED_SLASHES);
    }
    $line .= "\n";

    @file_put_contents(QUEUE_WORKER_LOG, $line, FILE_APPEND);
}

function claim_next_job($pdo)
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
    } catch (Exception $error) {
        $pdo->rollBack();
        throw $error;
    }

    return find_job($pdo, (int) $row['id']);
}

function mark_job_completed($pdo, $jobId, $graphId, $manifest)
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

function mark_job_failed($pdo, $jobId, $message)
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

function powershell_literal($value)
{
    return "'" . str_replace("'", "''", (string) $value) . "'";
}

function build_remote_command($job)
{
    $remoteScript = str_replace('\\', '/', REMOTE_REPO_ROOT) . '/web/tools/run_remote_prediction_job.py';
    $parts = [
        powershell_literal(REMOTE_PYTHON),
        powershell_literal($remoteScript),
        powershell_literal('--ticker'),
        powershell_literal((string) $job['requested_ticker']),
        powershell_literal('--job-id'),
        powershell_literal((string) $job['id']),
        powershell_literal('--device'),
        powershell_literal((string) $job['requested_device']),
        powershell_literal('--epochs'),
        powershell_literal((string) $job['requested_epochs']),
        powershell_literal('--batch-size'),
        powershell_literal((string) $job['requested_batch_size']),
        powershell_literal('--prediction-days'),
        powershell_literal((string) $job['requested_prediction_days']),
        powershell_literal('--future-days'),
        powershell_literal((string) $job['requested_future_days']),
        powershell_literal('--mc-runs'),
        powershell_literal((string) $job['requested_mc_runs']),
        powershell_literal('--output-root'),
        powershell_literal(REMOTE_OUTPUT_ROOT),
    ];

    $psCommand = '& ' . implode(' ', $parts);
    return 'powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "' .
        str_replace('"', '\"', $psCommand) .
        '"';
}

function remote_training_config_issues()
{
    $issues = [];

    if (REMOTE_SSH_HOST === '' || strpos(REMOTE_SSH_HOST, 'YOUR_') !== false) {
        $issues[] = 'REMOTE_SSH_HOST still needs the Windows PC host/IP.';
    }

    if (REMOTE_SSH_USER === '' || strpos(REMOTE_SSH_USER, 'YOUR_') !== false) {
        $issues[] = 'REMOTE_SSH_USER still needs the Windows SSH username.';
    }

    if (REMOTE_SSH_KEY === '' || strpos(REMOTE_SSH_KEY, 'your-linux-user') !== false) {
        $issues[] = 'REMOTE_SSH_KEY still needs the real Linux private key path.';
    } elseif (!is_file(REMOTE_SSH_KEY)) {
        $issues[] = 'REMOTE_SSH_KEY does not exist at ' . REMOTE_SSH_KEY . '.';
    }

    if (REMOTE_REPO_ROOT === '') {
        $issues[] = 'REMOTE_REPO_ROOT is empty.';
    }

    if (REMOTE_PYTHON === '') {
        $issues[] = 'REMOTE_PYTHON is empty.';
    }

    if (REMOTE_OUTPUT_ROOT === '') {
        $issues[] = 'REMOTE_OUTPUT_ROOT is empty.';
    }

    return $issues;
}

function build_ssh_training_command($job)
{
    $sshTarget = REMOTE_SSH_USER . '@' . REMOTE_SSH_HOST;
    $remoteCommand = build_remote_command($job);

    return sprintf(
        'ssh -i %s -p %d %s %s',
        escapeshellarg(REMOTE_SSH_KEY),
        REMOTE_SSH_PORT,
        escapeshellarg($sshTarget),
        escapeshellarg($remoteCommand)
    );
}

function run_remote_training_job($job)
{
    $configIssues = remote_training_config_issues();
    if ($configIssues !== []) {
        throw new RuntimeException('Remote training is not configured: ' . implode(' ', $configIssues));
    }

    $command = build_ssh_training_command($job);
    log_queue_worker('Running remote training command.', ['job_id' => (int) $job['id'], 'command' => $command]);

    $output = [];
    $exitCode = 0;
    exec($command . ' 2>&1', $output, $exitCode);
    $raw = trim(implode("\n", $output));

    if ($exitCode !== 0) {
        throw new RuntimeException('Remote training failed with exit code ' . (string) $exitCode . ': ' . ($raw !== '' ? $raw : '[no output]'));
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

function normalise_remote_scp_path($path)
{
    $path = str_replace('\\', '/', $path);
    if (preg_match('/^[A-Za-z]:\//', $path) === 1) {
        return '/' . strtoupper($path[0]) . substr($path, 1);
    }
    return $path;
}

function scp_remote_file($remotePath, $localPath)
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

function import_remote_manifest($manifest, $job)
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
        'residuals_plot' => $jobDir . '/residuals.png',
        'predictions_csv' => $jobDir . '/predictions.csv',
        'forecast_csv' => $jobDir . '/forecast.csv',
        'training_log' => $jobDir . '/training.log',
        'manifest_json' => $jobDir . '/manifest.json',
    ];

    $remoteFiles = isset($manifest['files']) && is_array($manifest['files']) ? $manifest['files'] : [];
    $required = [
        'summary_plot' => $local['summary_plot'],
        'detail_plot' => $local['detail_plot'],
        'residuals_plot' => $local['residuals_plot'],
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

function save_graph_from_job($pdo, $job, $manifest, $imports)
{
    $summary = isset($manifest['summary']) ? $manifest['summary'] : null;
    $stats = isset($manifest['stats']) && is_array($manifest['stats']) ? $manifest['stats'] : [];

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
    $stmt->bindValue('summary_text', (string) ($summary !== null ? $summary : 'Queued training result imported from the remote forecasting machine.'));
    $stmt->bindValue('summary_plot_path', $imports['summary_plot']);
    $stmt->bindValue('detail_plot_path', $imports['detail_plot']);
    $stmt->bindValue('predictions_csv_path', $imports['predictions_csv']);
    $stmt->bindValue('forecast_csv_path', $imports['forecast_csv']);
    $stmt->bindValue('remote_job_directory', isset($manifest['job_dir']) ? (string) $manifest['job_dir'] : '');
    $stmt->execute();

    $graphId = (int) $pdo->lastInsertId();

    foreach (['summary' => $imports['summary_plot'], 'detail' => $imports['detail_plot'], 'residuals' => $imports['residuals_plot']] as $kind => $path) {
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

        if (isset($stats['residual_mean']) || isset($stats['residual_std'])) {
            $residualStmt = $pdo->prepare(
                "UPDATE saved_graphs
                 SET summary_text = CONCAT(summary_text, '\nResidual mean: ', :residual_mean, ' | Residual standard deviation: ', :residual_std)
                 WHERE id = :graph_id"
            );
            $residualStmt->execute([
                'residual_mean' => isset($stats['residual_mean']) ? (string) $stats['residual_mean'] : 'n/a',
                'residual_std' => isset($stats['residual_std']) ? (string) $stats['residual_std'] : 'n/a',
                'graph_id' => $graphId,
            ]);
        }
    }

    return $graphId;
}
