<?php

require_once __DIR__ . '/includes/bootstrap.php';

$dbError = null;
$pdo = db_optional($dbError);

$stats = [
    'queued_jobs' => 0,
    'running_jobs' => 0,
    'saved_graphs' => 0,
    'tracked_tickers' => 0,
];
$recentJobs = [];
$recentGraphs = [];

if ($pdo instanceof PDO) {
    $stats = fetch_dashboard_stats($pdo);
    $recentJobs = fetch_recent_jobs($pdo, 5);
    $recentGraphs = fetch_recent_graphs($pdo, 4);
}

render_layout_start('Home', 'home');
?>

<h1>Forecast queue and graph archive</h1>
<p>
    This site lets a user log in, search for a ticker, queue a new training run, and reopen older saved graphs.
    PHP handles the web flow, MySQL stores the records, and the queue worker is designed to start one remote training job at a time.
</p>

<?php if ($dbError !== null): ?>
    <section>
        <p>
            <strong>Warning:</strong>
            The website loaded, but the database is not connected yet. You can still test the login and page flow,
            but queueing forecasts and loading saved graphs will not work until the database is configured.
        </p>
    </section>
<?php endif; ?>

<section>
    <h2>Quick links</h2>
    <p>
        <a href="<?= h(app_url('/search.php')) ?>">Open Search Page</a> |
        <a href="<?= h(app_url('/login.php')) ?>">Log In</a> |
        <a href="<?= h(app_url('/register.php')) ?>">Create Account</a>
    </p>
</section>

<section>
    <h2>Current totals</h2>
    <ul>
        <li>Queued jobs: <?= h((string) $stats['queued_jobs']) ?></li>
        <li>Running jobs: <?= h((string) $stats['running_jobs']) ?></li>
        <li>Saved graphs: <?= h((string) $stats['saved_graphs']) ?></li>
        <li>Tracked tickers: <?= h((string) $stats['tracked_tickers']) ?></li>
    </ul>
</section>

<section>
    <h2>How the site works</h2>
    <ol>
        <li>Log in and search for a ticker.</li>
        <li>Open an older saved graph or request a new forecast.</li>
        <li>The request is written into the `prediction_jobs` queue table.</li>
        <li>A PHP worker starts a single remote training run over SSH.</li>
        <li>When the run finishes, the generated graph files are imported back into the site and stored in the database archive.</li>
    </ol>
</section>

<section>
    <h2>Demo account</h2>
    <p>Username: <?= h(DEMO_USERNAME) ?></p>
    <p>Password: <?= h(DEMO_PASSWORD) ?></p>
</section>

<section>
    <h2>Recent jobs</h2>
    <?php if ($recentJobs === []): ?>
        <p>No jobs have been queued yet.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($recentJobs as $job): ?>
                <li>
                    <a href="<?= h(app_url('/view.php?job=' . (int) $job['id'])) ?>">
                        <?= h($job['ticker_symbol']) ?> - <?= h($job['status']) ?> - <?= h((string) $job['created_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<section>
    <h2>Recent graphs</h2>
    <?php if ($recentGraphs === []): ?>
        <p>No graph images have been imported yet.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($recentGraphs as $graph): ?>
                <li>
                    <a href="<?= h(app_url('/view.php?graph=' . (int) $graph['id'])) ?>">
                        <?= h($graph['ticker_symbol']) ?> - <?= h($graph['title']) ?> - <?= h((string) $graph['created_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<?php render_layout_end(); ?>
