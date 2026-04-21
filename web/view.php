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
    <section class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Queued Forecast</p>
                <h1>Job #<?= h((string) $job['id']) ?> for <?= h($job['ticker_symbol']) ?></h1>
            </div>
            <span class="status-pill status-<?= h((string) $job['status']) ?>"><?= h((string) $job['status']) ?></span>
        </div>

        <div class="info-grid">
            <div class="info-item">
                <span>Requested</span>
                <strong><?= h((string) $job['created_at']) ?></strong>
            </div>
            <div class="info-item">
                <span>Device</span>
                <strong><?= h((string) $job['requested_device']) ?></strong>
            </div>
            <div class="info-item">
                <span>Epochs</span>
                <strong><?= h((string) $job['requested_epochs']) ?></strong>
            </div>
            <div class="info-item">
                <span>Future Days</span>
                <strong><?= h((string) $job['requested_future_days']) ?></strong>
            </div>
            <?php if ($job['status'] === 'queued'): ?>
                <div class="info-item">
                    <span>Queue Position</span>
                    <strong><?= h((string) (($position = queue_position(db(), (int) $job['id'])) !== null ? $position : 1)) ?></strong>
                </div>
            <?php endif; ?>
        </div>

        <?php if (!empty($job['failure_message'])): ?>
            <div class="flash flash-danger"><?= h((string) $job['failure_message']) ?></div>
        <?php elseif (!empty($job['output_message'])): ?>
            <div class="flash flash-success"><?= h((string) $job['output_message']) ?></div>
        <?php endif; ?>

        <?php if (!empty($job['saved_graph_id'])): ?>
            <div class="hero-actions">
                <a class="button button-primary" href="<?= h(app_url('/view.php?graph=' . (int) $job['saved_graph_id'])) ?>">Open Finished Graph</a>
            </div>
        <?php endif; ?>
    </section>
<?php endif; ?>

<?php if ($graph !== null): ?>
    <section class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Saved Graph</p>
                <h1><?= h($graph['title']) ?></h1>
            </div>
            <span class="status-pill status-completed">archived</span>
        </div>
        <p class="support-copy"><?= nl2br(h((string) $graph['summary_text'])) ?></p>

        <div class="graph-grid">
            <article class="graph-card">
                <h2>Primary View</h2>
                <?php if (graph_asset_exists(db(), (int) $graph['id'], 'summary')): ?>
                    <img src="<?= h(app_url('/asset.php?graph=' . (int) $graph['id'] . '&kind=summary')) ?>" alt="Summary graph for <?= h($graph['ticker_symbol']) ?>">
                <?php else: ?>
                    <p class="empty-state">Summary graph asset is not available yet.</p>
                <?php endif; ?>
            </article>
            <article class="graph-card">
                <h2>Forecast Detail</h2>
                <?php if (graph_asset_exists(db(), (int) $graph['id'], 'detail')): ?>
                    <img src="<?= h(app_url('/asset.php?graph=' . (int) $graph['id'] . '&kind=detail')) ?>" alt="Forecast detail graph for <?= h($graph['ticker_symbol']) ?>">
                <?php else: ?>
                    <p class="empty-state">Detail graph asset is not available yet.</p>
                <?php endif; ?>
            </article>
        </div>

        <div class="hero-actions">
            <form method="post" action="<?= h(app_url('/request_prediction.php')) ?>">
                <input type="hidden" name="ticker" value="<?= h($graph['ticker_symbol']) ?>">
                <button class="button button-primary" type="submit">Create New Graph</button>
            </form>
            <a class="button button-secondary" href="<?= h(app_url('/search.php?ticker=' . urlencode((string) $graph['ticker_symbol']))) ?>">Back to Search</a>
        </div>
    </section>
<?php endif; ?>

<section class="content-grid two-up">
    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Ticker History</p>
                <h2>Previous saved graphs</h2>
            </div>
        </div>
        <?php if ($tickerGraphs === []): ?>
            <p class="empty-state">There are no saved graphs for this ticker yet.</p>
        <?php else: ?>
            <div class="stack-list">
                <?php foreach ($tickerGraphs as $item): ?>
                    <a class="list-card" href="<?= h(app_url('/view.php?graph=' . (int) $item['id'])) ?>">
                        <div>
                            <strong><?= h($item['title']) ?></strong>
                            <span><?= h($item['ticker_symbol']) ?></span>
                        </div>
                        <small><?= h((string) $item['created_at']) ?></small>
                    </a>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </article>

    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Ticker Queue</p>
                <h2>Recent jobs for this symbol</h2>
            </div>
        </div>
        <?php if ($tickerJobs === []): ?>
            <p class="empty-state">No jobs have been recorded for this ticker yet.</p>
        <?php else: ?>
            <div class="stack-list">
                <?php foreach ($tickerJobs as $item): ?>
                    <a class="list-card" href="<?= h(app_url('/view.php?job=' . (int) $item['id'])) ?>">
                        <div>
                            <strong>Job #<?= h((string) $item['id']) ?></strong>
                            <span><?= h($item['status']) ?></span>
                        </div>
                        <small><?= h((string) $item['created_at']) ?></small>
                    </a>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </article>
</section>

<?php render_layout_end(); ?>
