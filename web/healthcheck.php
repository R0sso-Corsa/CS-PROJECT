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

<h1>Server health check</h1>
<p>Use this page to confirm that the Linux/XAMPP deployment is executing PHP correctly, preserving sessions, writing to storage, and reaching the configured database.</p>

<section>
    <h2>Application status</h2>
    <ul>
        <?php foreach ($checks as $check): ?>
            <li>
                <strong><?= h($check['label']) ?></strong>:
                <?= $check['ok'] ? 'ok' : 'failed' ?> -
                <?= h($check['detail']) ?>
            </li>
        <?php endforeach; ?>
    </ul>
</section>

<section>
    <h2>Runtime values</h2>
    <ul>
        <li>Current URL base: <?= h(app_base_path() === '' ? '/' : app_base_path()) ?></li>
        <li>Script name: <?= h(isset($_SERVER['SCRIPT_NAME']) ? (string) $_SERVER['SCRIPT_NAME'] : 'unknown') ?></li>
        <li>Request URI: <?= h(isset($_SERVER['REQUEST_URI']) ? (string) $_SERVER['REQUEST_URI'] : 'unknown') ?></li>
        <li>Signed-in user: <?= h(isset($user['username']) ? $user['username'] : 'No active session user') ?></li>
    </ul>
</section>

<section>
    <h2>Storage folder status</h2>
    <ul>
        <?php foreach ($storageResults as $label => $result): ?>
            <li>
                <strong><?= h($label) ?></strong>:
                <?= h($result['path']) ?> -
                <?= h($result['exists'] ? 'exists' : 'missing') ?> /
                <?= h($result['writable'] ? 'writable' : 'not writable') ?>
            </li>
        <?php endforeach; ?>
    </ul>
</section>

<?php render_layout_end(); ?>
