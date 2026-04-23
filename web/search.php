<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$dbError = null;
$pdo = db_optional($dbError);
$query = isset($_GET['ticker']) ? normalise_ticker((string) $_GET['ticker']) : '';
$matches = [];
$existingGraphs = [];
$jobHistory = [];
$recentGraphs = [];
$recentJobs = [];

if ($pdo instanceof PDO) {
    $matches = $query !== '' ? search_ticker_history($pdo, $query) : [];
    $existingGraphs = $query !== '' ? fetch_graphs_for_ticker($pdo, $query) : [];
    $jobHistory = $query !== '' ? fetch_jobs_for_ticker($pdo, $query) : [];
    $recentGraphs = $query === '' ? fetch_recent_graphs($pdo, 10) : [];
    $recentJobs = fetch_recent_jobs($pdo, 8);
}

render_layout_start('Search', 'search');
?>

<h1>Search ticker history</h1>
<p>Search a symbol, inspect any stored history, and then choose an older saved graph or queue a new one.</p>

<?php if ($dbError !== null): ?>
    <section>
        <p><strong>Warning:</strong> You are signed in, but the database is not connected yet. Search history, queueing, and saved graphs are disabled until the database is configured.</p>
    </section>
<?php endif; ?>

<section>
    <h2>Search form</h2>
    <form method="get">
        <p>
            <label>
                Ticker<br>
                <input
                    type="text"
                    name="ticker"
                    value="<?= h($query) ?>"
                    placeholder="BTC-USD, ETH-USD, AAPL, TSLA"
                    required
                >
            </label>
            <button type="submit">Search</button>
        </p>
    </form>
    <p>Quick fill:</p>
    <p>
        <button data-fill-ticker="BTC-USD" type="button">BTC-USD</button>
        <button data-fill-ticker="ETH-USD" type="button">ETH-USD</button>
        <button data-fill-ticker="SOL-USD" type="button">SOL-USD</button>
        <button data-fill-ticker="AAPL" type="button">AAPL</button>
        <button data-fill-ticker="TSLA" type="button">TSLA</button>
    </p>
</section>

<?php if ($query !== ''): ?>
    <section>
        <h2>Queue a fresh forecast for <?= h($query) ?></h2>
        <p>Submitting a new request adds the ticker to the queue if another training run is already active.</p>
        <?php if ($pdo instanceof PDO): ?>
            <form method="post" action="<?= h(app_url('/request_prediction.php')) ?>">
                <input type="hidden" name="ticker" value="<?= h($query) ?>">
                <button type="submit">Create New Graph</button>
            </form>
        <?php else: ?>
            <p>Queueing is disabled until the database connection is working.</p>
        <?php endif; ?>
    </section>

    <section>
        <h2>Matches for <?= h($query) ?></h2>
        <?php if ($matches === []): ?>
            <p>This ticker is not in the database yet, but you can still queue a new forecast for it.</p>
        <?php else: ?>
            <ul>
                <?php foreach ($matches as $match): ?>
                    <li>
                        <?= h($match['symbol']) ?> -
                        <?= h((string) $match['graph_count']) ?> saved graphs -
                        <?= h(isset($match['latest_graph_at']) ? (string) $match['latest_graph_at'] : 'No graph imported yet') ?>
                    </li>
                <?php endforeach; ?>
            </ul>
        <?php endif; ?>
    </section>

    <section>
        <h2>Previous graphs</h2>
        <?php if ($existingGraphs === []): ?>
            <p>There are no saved graphs for this ticker yet.</p>
        <?php else: ?>
            <ul>
                <?php foreach ($existingGraphs as $graph): ?>
                    <li>
                        <a href="<?= h(app_url('/view.php?graph=' . (int) $graph['id'])) ?>">
                            <?= h($graph['title']) ?> - <?= h($graph['ticker_symbol']) ?> - <?= h((string) $graph['created_at']) ?>
                        </a>
                    </li>
                <?php endforeach; ?>
            </ul>
        <?php endif; ?>
    </section>

    <section>
        <h2>Job history for <?= h($query) ?></h2>
        <?php if ($jobHistory === []): ?>
            <p>No jobs have been queued for this ticker yet.</p>
        <?php else: ?>
            <ul>
                <?php foreach ($jobHistory as $job): ?>
                    <li>
                        <a href="<?= h(app_url('/view.php?job=' . (int) $job['id'])) ?>">
                            Job #<?= h((string) $job['id']) ?> - <?= h($job['status']) ?> - <?= h((string) $job['created_at']) ?>
                        </a>
                    </li>
                <?php endforeach; ?>
            </ul>
        <?php endif; ?>
    </section>
<?php else: ?>
    <section>
        <h2>Recently imported graphs</h2>
        <?php if ($recentGraphs === []): ?>
            <p>No graphs are stored yet. Queue your first run from the search box above.</p>
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

    <section>
        <h2>Current job list</h2>
        <?php if ($recentJobs === []): ?>
            <p>Nothing is in the queue yet.</p>
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
<?php endif; ?>

<?php render_layout_end(); ?>
