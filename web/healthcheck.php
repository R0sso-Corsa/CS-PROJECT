<?php

require_once __DIR__ . '/includes/bootstrap.php';

$dbError = null;
$pdo = db_optional($dbError);
$user = current_user();

$sessionSavePath = session_save_path();
$sessionWritable = $sessionSavePath !== '' && is_dir($sessionSavePath) && is_writable($sessionSavePath);

$storageChecks = [
    'storage_root' => WEB_STORAGE_ROOT,
    'graphs_root' => GRAPH_IMPORT_ROOT,
    'logs_root' => LOG_ROOT,
];

$storageResults = [];
foreach ($storageChecks as $label => $path) {
    $storageResults[$label] = [
        'path' => $path,
        'exists' => is_dir($path),
        'writable' => is_dir($path) && is_writable($path),
    ];
}

$tempProbePath = LOG_ROOT . '/healthcheck_probe.tmp';
$storageProbeResult = false;
$storageProbeMessage = 'Skipped';

try {
    $bytes = @file_put_contents($tempProbePath, 'healthcheck ' . date(DATE_ATOM));
    if ($bytes === false) {
        $storageProbeMessage = 'Failed to write probe file.';
    } else {
        $storageProbeResult = true;
        $storageProbeMessage = 'Probe file written successfully.';
        @unlink($tempProbePath);
    }
} catch (Exception $exception) {
    $storageProbeMessage = 'Exception while writing probe file: ' . $exception->getMessage();
}

$checks = [
    [
        'label' => 'PHP is executing',
        'ok' => true,
        'detail' => 'PHP ' . PHP_VERSION,
    ],
    [
        'label' => 'Session started',
        'ok' => session_status() === PHP_SESSION_ACTIVE,
        'detail' => 'Session ID: ' . session_id(),
    ],
    [
        'label' => 'Session save path writable',
        'ok' => $sessionWritable,
        'detail' => $sessionSavePath !== '' ? $sessionSavePath : 'session_save_path() is empty',
    ],
    [
        'label' => 'Storage write probe',
        'ok' => $storageProbeResult,
        'detail' => $storageProbeMessage,
    ],
    [
        'label' => 'Database connection',
        'ok' => $pdo instanceof PDO,
        'detail' => $pdo instanceof PDO ? 'Connected to ' . DB_NAME : ($dbError !== null ? $dbError : 'Unknown database error'),
    ],
];

render_layout_start('Health Check', 'home');
?>

<section class="panel">
    <div class="panel-heading">
        <div>
            <p class="eyebrow">Diagnostics</p>
            <h1>Server health check</h1>
        </div>
    </div>
    <p class="support-copy">
        Use this page to confirm that the Linux/XAMPP deployment is executing PHP correctly, preserving sessions, writing to the site storage folders, and reaching the configured database.
    </p>
</section>

<section class="content-grid two-up">
    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Core Checks</p>
                <h2>Application status</h2>
            </div>
        </div>
        <div class="stack-list">
            <?php foreach ($checks as $check): ?>
                <div class="list-card list-card-static">
                    <div>
                        <strong><?= h($check['label']) ?></strong>
                        <span><?= h($check['detail']) ?></span>
                    </div>
                    <span class="status-pill <?= $check['ok'] ? 'status-completed' : 'status-failed' ?>">
                        <?= $check['ok'] ? 'ok' : 'failed' ?>
                    </span>
                </div>
            <?php endforeach; ?>
        </div>
    </article>

    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Request Context</p>
                <h2>Runtime values</h2>
            </div>
        </div>
        <div class="stack-list">
            <div class="list-card list-card-static">
                <div>
                    <strong>Current URL base</strong>
                    <span><?= h(app_base_path() === '' ? '/' : app_base_path()) ?></span>
                </div>
            </div>
            <div class="list-card list-card-static">
                <div>
                    <strong>Script name</strong>
                    <span><?= h(isset($_SERVER['SCRIPT_NAME']) ? (string) $_SERVER['SCRIPT_NAME'] : 'unknown') ?></span>
                </div>
            </div>
            <div class="list-card list-card-static">
                <div>
                    <strong>Request URI</strong>
                    <span><?= h(isset($_SERVER['REQUEST_URI']) ? (string) $_SERVER['REQUEST_URI'] : 'unknown') ?></span>
                </div>
            </div>
            <div class="list-card list-card-static">
                <div>
                    <strong>Signed-in user</strong>
                    <span><?= h(isset($user['username']) ? $user['username'] : 'No active session user') ?></span>
                </div>
            </div>
        </div>
    </article>
</section>

<section class="panel">
    <div class="panel-heading">
        <div>
            <p class="eyebrow">Filesystem</p>
            <h2>Storage folder status</h2>
        </div>
    </div>
    <div class="stack-list">
        <?php foreach ($storageResults as $label => $result): ?>
            <div class="list-card list-card-static">
                <div>
                    <strong><?= h($label) ?></strong>
                    <span><?= h($result['path']) ?></span>
                </div>
                <small>
                    <?= h($result['exists'] ? 'exists' : 'missing') ?> /
                    <?= h($result['writable'] ? 'writable' : 'not writable') ?>
                </small>
            </div>
        <?php endforeach; ?>
    </div>
</section>

<?php render_layout_end(); ?>
