<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$graphId = isset($_GET['graph']) ? (int) $_GET['graph'] : null;
$jobId = isset($_GET['job']) ? (int) $_GET['job'] : null;

$graph = $graphId ? find_graph(db(), $graphId) : null;
$job = $jobId ? find_job(db(), $jobId) : null;

if ($graph === null && $job === null) {
    set_flash('That graph or job could not be found.', 'danger');
    header('Location: ' . app_url('/search.php'));
    exit;
}

if ($graph !== null) {
    $tickerGraphs = fetch_graphs_for_ticker(db(), (string) $graph['ticker_symbol'], 6);
    $tickerJobs = fetch_jobs_for_ticker(db(), (string) $graph['ticker_symbol'], 6);
} else {
    $tickerGraphs = fetch_graphs_for_ticker(db(), (string) $job['ticker_symbol'], 6);
    $tickerJobs = fetch_jobs_for_ticker(db(), (string) $job['ticker_symbol'], 6);
}

render_layout_start('View', 'search');
?>

<?php if ($job !== null): ?>
    <section>
        <h1>Job #<?= h((string) $job['id']) ?> for <?= h($job['ticker_symbol']) ?></h1>
        <p>Status: <?= h((string) $job['status']) ?></p>
        <dl>
            <dt>Requested</dt>
            <dd><?= h((string) $job['created_at']) ?></dd>
            <dt>Device</dt>
            <dd><?= h((string) $job['requested_device']) ?></dd>
            <dt>Epochs</dt>
            <dd><?= h((string) $job['requested_epochs']) ?></dd>
            <dt>Future days</dt>
            <dd><?= h((string) $job['requested_future_days']) ?></dd>
            <?php if ($job['status'] === 'queued'): ?>
                <dt>Queue position</dt>
                <dd><?= h((string) (($position = queue_position(db(), (int) $job['id'])) !== null ? $position : 1)) ?></dd>
            <?php endif; ?>
        </dl>

        <?php if (!empty($job['failure_message'])): ?>
            <section>
                <p><strong>Error:</strong> <?= h((string) $job['failure_message']) ?></p>
            </section>
        <?php elseif (!empty($job['output_message'])): ?>
            <section>
                <p><strong>Update:</strong> <?= h((string) $job['output_message']) ?></p>
            </section>
        <?php endif; ?>

        <?php if (!empty($job['saved_graph_id'])): ?>
            <p><a href="<?= h(app_url('/view.php?graph=' . (int) $job['saved_graph_id'])) ?>">Open Finished Graph</a></p>
        <?php endif; ?>
    </section>
<?php endif; ?>

<?php if ($graph !== null): ?>
    <section>
        <h1><?= h($graph['title']) ?></h1>
        <p>Ticker: <?= h($graph['ticker_symbol']) ?></p>
        <p><?= nl2br(h((string) $graph['summary_text'])) ?></p>

        <section>
            <h2>Primary view</h2>
            <?php if (graph_asset_exists(db(), (int) $graph['id'], 'summary')): ?>
                <img src="<?= h(app_url('/asset.php?graph=' . (int) $graph['id'] . '&kind=summary')) ?>" alt="Summary graph for <?= h($graph['ticker_symbol']) ?>">
            <?php else: ?>
                <p>Summary graph asset is not available yet.</p>
            <?php endif; ?>
        </section>

        <section>
            <h2>Forecast detail</h2>
            <?php if (graph_asset_exists(db(), (int) $graph['id'], 'detail')): ?>
                <img src="<?= h(app_url('/asset.php?graph=' . (int) $graph['id'] . '&kind=detail')) ?>" alt="Forecast detail graph for <?= h($graph['ticker_symbol']) ?>">
            <?php else: ?>
                <p>Detail graph asset is not available yet.</p>
            <?php endif; ?>
        </section>

        <section>
            <h2>Residuals</h2>
            <?php if (graph_asset_exists(db(), (int) $graph['id'], 'residuals')): ?>
                <img src="<?= h(app_url('/asset.php?graph=' . (int) $graph['id'] . '&kind=residuals')) ?>" alt="Residuals graph for <?= h($graph['ticker_symbol']) ?>">
            <?php else: ?>
                <p>Residuals graph asset is not available yet.</p>
            <?php endif; ?>
        </section>

        <form method="post" action="<?= h(app_url('/request_prediction.php')) ?>">
            <input type="hidden" name="ticker" value="<?= h($graph['ticker_symbol']) ?>">
            <p><button type="submit">Create New Graph</button></p>
        </form>
        <p><a href="<?= h(app_url('/search.php?ticker=' . urlencode((string) $graph['ticker_symbol']))) ?>">Back to Search</a></p>
    </section>
<?php endif; ?>

<section>
    <h2>Previous saved graphs</h2>
    <?php if ($tickerGraphs === []): ?>
        <p>There are no saved graphs for this ticker yet.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($tickerGraphs as $item): ?>
                <li>
                    <a href="<?= h(app_url('/view.php?graph=' . (int) $item['id'])) ?>">
                        <?= h($item['title']) ?> - <?= h($item['ticker_symbol']) ?> - <?= h((string) $item['created_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<section>
    <h2>Recent jobs for this ticker</h2>
    <?php if ($tickerJobs === []): ?>
        <p>No jobs have been recorded for this ticker yet.</p>
    <?php else: ?>
        <ul>
            <?php foreach ($tickerJobs as $item): ?>
                <li>
                    <a href="<?= h(app_url('/view.php?job=' . (int) $item['id'])) ?>">
                        Job #<?= h((string) $item['id']) ?> - <?= h($item['status']) ?> - <?= h((string) $item['created_at']) ?>
                    </a>
                </li>
            <?php endforeach; ?>
        </ul>
    <?php endif; ?>
</section>

<?php render_layout_end(); ?>
