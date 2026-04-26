<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$dbError = null;
$pdo = db_optional($dbError);
$jobId = isset($_GET['job']) ? (int) $_GET['job'] : 0;
$job = $pdo instanceof PDO && $jobId > 0 ? find_job($pdo, $jobId) : null;
$configIssues = remote_training_config_issues();
$recentJobs = $pdo instanceof PDO ? fetch_recent_jobs($pdo, 8) : [];
$queueCounts = $pdo instanceof PDO ? fetch_queue_status_counts($pdo) : [];
$runningJobs = $pdo instanceof PDO ? fetch_running_jobs($pdo, 5) : [];
$command = $job !== null ? build_ssh_training_command($job) : '';
$remoteCommand = $job !== null ? build_remote_command($job) : '';
$logTail = '';

if (is_file(QUEUE_WORKER_LOG) && is_readable(QUEUE_WORKER_LOG)) {
    $lines = file(QUEUE_WORKER_LOG, FILE_IGNORE_NEW_LINES);
    if (is_array($lines)) {
        $logTail = implode("\n", array_slice($lines, -80));
    }
}

$checks = [
    'PHP CLI binary' => [
        'value' => PHP_CLI_BINARY,
        'ok' => is_file(PHP_CLI_BINARY) || stripos(PHP_CLI_BINARY, 'php') !== false,
    ],
    'Queue worker script' => [
        'value' => QUEUE_WORKER_SCRIPT,
        'ok' => is_file(QUEUE_WORKER_SCRIPT),
    ],
    'Queue log folder writable' => [
        'value' => LOG_ROOT,
        'ok' => is_dir(LOG_ROOT) && is_writable(LOG_ROOT),
    ],
    'Queue lock path' => [
        'value' => QUEUE_WORKER_LOCK,
        'ok' => is_dir(dirname(QUEUE_WORKER_LOCK)) && is_writable(dirname(QUEUE_WORKER_LOCK)),
    ],
    'SSH key' => [
        'value' => REMOTE_SSH_KEY,
        'ok' => is_file(REMOTE_SSH_KEY),
    ],
    'SSH known hosts' => [
        'value' => remote_ssh_known_hosts_path(),
        'ok' => is_file(remote_ssh_known_hosts_path()) && is_readable(remote_ssh_known_hosts_path()),
    ],
];

render_layout_start('Queue Debug', 'search');
?>

<h1>Queue debug</h1>
<p>This page checks the queue worker configuration used when a forecast job is executed.</p>

<?php if ($dbError !== null): ?>
    <section>
        <p><strong>Database error:</strong> <?= h($dbError) ?></p>
    </section>
<?php endif; ?>

<section>
    <h2>Configuration issues</h2>
    <?php if ($configIssues === []): ?>
        <p>No obvious remote training configuration problems were found.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($configIssues as $issue): ?>
                <li><?= h($issue) ?></li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<section>
    <h2>Local worker checks</h2>
    <dl>
        <?php foreach ($checks as $label => $check): ?>
            <dt><?= h($label) ?></dt>
            <dd><?= h($check['value']) ?> - <?= $check['ok'] ? 'ok' : 'problem' ?></dd>
        <?php endforeach; ?>
    </dl>
</section>

<section>
    <h2>Remote settings</h2>
    <dl>
        <dt>REMOTE_SSH_HOST</dt>
        <dd><?= h(REMOTE_SSH_HOST) ?></dd>
        <dt>REMOTE_SSH_PORT</dt>
        <dd><?= h((string) REMOTE_SSH_PORT) ?></dd>
        <dt>REMOTE_SSH_USER</dt>
        <dd><?= h(REMOTE_SSH_USER) ?></dd>
        <dt>SSH known_hosts path</dt>
        <dd><?= h(remote_ssh_known_hosts_path()) ?></dd>
        <dt>REMOTE_REPO_ROOT</dt>
        <dd><?= h(REMOTE_REPO_ROOT) ?></dd>
        <dt>REMOTE_PYTHON</dt>
        <dd><?= h(REMOTE_PYTHON) ?></dd>
        <dt>REMOTE_OUTPUT_ROOT</dt>
        <dd><?= h(REMOTE_OUTPUT_ROOT) ?></dd>
    </dl>
</section>

<section>
    <h2>Queue status</h2>
    <?php if ($queueCounts === []): ?>
        <p>Queue counts are unavailable because the database is not connected.</p>
    <?php else: ?>
        <dl>
            <dt>Queued</dt>
            <dd><?= h((string) $queueCounts['queued']) ?></dd>
            <dt>Running</dt>
            <dd><?= h((string) $queueCounts['running']) ?></dd>
            <dt>Completed</dt>
            <dd><?= h((string) $queueCounts['completed']) ?></dd>
            <dt>Failed</dt>
            <dd><?= h((string) $queueCounts['failed']) ?></dd>
        </dl>
    <?php endif; ?>

    <?php if ($runningJobs !== []): ?>
        <p><strong>Important:</strong> a running job blocks the queue from starting another job.</p>
        <ul>
            <?php foreach ($runningJobs as $running): ?>
                <li>
                    <a href="<?= h(app_url('/queue_debug.php?job=' . (int) $running['id'])) ?>">
                        Job #<?= h((string) $running['id']) ?> - <?= h($running['ticker_symbol']) ?> - started <?= h((string) $running['started_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<section>
    <h2>Inspect job command</h2>
    <form method="get">
        <p>
            <label>
                Job ID<br>
                <input type="number" name="job" value="<?= h($jobId > 0 ? (string) $jobId : '') ?>">
            </label>
            <button type="submit">Show command</button>
        </p>
    </form>

    <?php if ($jobId > 0 && $job === null): ?>
        <p>No job was found for that ID.</p>
    <?php elseif ($job !== null): ?>
        <dl>
            <dt>Job</dt>
            <dd>#<?= h((string) $job['id']) ?> - <?= h((string) $job['ticker_symbol']) ?> - <?= h((string) $job['status']) ?></dd>
            <dt>Remote PowerShell command</dt>
            <dd><pre><?= h($remoteCommand) ?></pre></dd>
            <dt>Full SSH command</dt>
            <dd><pre><?= h($command) ?></pre></dd>
            <?php if (!empty($job['failure_message'])): ?>
                <dt>Failure message</dt>
                <dd><pre><?= h((string) $job['failure_message']) ?></pre></dd>
            <?php endif; ?>
        </dl>
    <?php endif; ?>
</section>

<section>
    <h2>Queue worker log tail</h2>
    <?php if ($logTail === ''): ?>
        <p>No readable queue worker log was found at <?= h(QUEUE_WORKER_LOG) ?>.</p>
    <?php else: ?>
        <pre><?= h($logTail) ?></pre>
    <?php endif; ?>
</section>

<section>
    <h2>Recent jobs</h2>
    <?php if ($recentJobs === []): ?>
        <p>No jobs found.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($recentJobs as $recent): ?>
                <li>
                    <a href="<?= h(app_url('/queue_debug.php?job=' . (int) $recent['id'])) ?>">
                        Job #<?= h((string) $recent['id']) ?> - <?= h($recent['ticker_symbol']) ?> - <?= h($recent['status']) ?> - <?= h((string) $recent['created_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<?php render_layout_end(); ?>
