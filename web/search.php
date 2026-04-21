<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$query = normalise_ticker((string) ($_GET['ticker'] ?? ''));
$matches = $query !== '' ? search_ticker_history(db(), $query) : [];
$existingGraphs = $query !== '' ? fetch_graphs_for_ticker(db(), $query) : [];
$jobHistory = $query !== '' ? fetch_jobs_for_ticker(db(), $query) : [];
$recentGraphs = $query === '' ? fetch_recent_graphs(db(), 10) : [];
$recentJobs = fetch_recent_jobs(db(), 8);

render_layout_start('Search', 'search');
?>

<section class="panel search-hero">
    <div class="panel-heading">
        <div>
            <p class="eyebrow">Ticker Search</p>
            <h1>Search a market symbol, then choose an older graph or queue a new one.</h1>
        </div>
    </div>

    <form class="search-form" method="get">
        <label class="grow-field">
            <span>Ticker</span>
            <input
                type="text"
                name="ticker"
                value="<?= h($query) ?>"
                placeholder="BTC-USD, ETH-USD, AAPL, TSLA"
                required
            >
        </label>
        <button class="button button-primary" type="submit">Search</button>
    </form>

    <div class="ticker-chip-row">
        <button class="chip-button" data-fill-ticker="BTC-USD" type="button">BTC-USD</button>
        <button class="chip-button" data-fill-ticker="ETH-USD" type="button">ETH-USD</button>
        <button class="chip-button" data-fill-ticker="SOL-USD" type="button">SOL-USD</button>
        <button class="chip-button" data-fill-ticker="AAPL" type="button">AAPL</button>
        <button class="chip-button" data-fill-ticker="TSLA" type="button">TSLA</button>
    </div>
</section>

<?php if ($query !== ''): ?>
    <section class="content-grid two-up">
        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Create New</p>
                    <h2>Queue a fresh forecast for <?= h($query) ?></h2>
                </div>
            </div>
            <p class="support-copy">
                Submitting a new request will add the ticker to the queue if another training run is active. The background worker only starts one remote training job at a time.
            </p>
            <form class="queue-form" method="post" action="<?= h(app_url('/request_prediction.php')) ?>">
                <input type="hidden" name="ticker" value="<?= h($query) ?>">
                <button class="button button-primary" type="submit">Create New Graph</button>
            </form>
        </article>

        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Ticker Summary</p>
                    <h2>Matches for <?= h($query) ?></h2>
                </div>
            </div>
            <?php if ($matches === []): ?>
                <p class="empty-state">This ticker is not in the database yet, but you can still queue a new forecast for it.</p>
            <?php else: ?>
                <div class="stack-list">
                    <?php foreach ($matches as $match): ?>
                        <div class="list-card list-card-static">
                            <div>
                                <strong><?= h($match['symbol']) ?></strong>
                                <span><?= h((string) $match['graph_count']) ?> saved graphs</span>
                            </div>
                            <small><?= h((string) ($match['latest_graph_at'] ?? 'No graph imported yet')) ?></small>
                        </div>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </article>
    </section>

    <section class="content-grid two-up">
        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Previous Graphs</p>
                    <h2>Open an older saved view</h2>
                </div>
            </div>
            <?php if ($existingGraphs === []): ?>
                <p class="empty-state">There are no saved graphs for this ticker yet.</p>
            <?php else: ?>
                <div class="stack-list">
                    <?php foreach ($existingGraphs as $graph): ?>
                        <a class="list-card" href="<?= h(app_url('/view.php?graph=' . (int) $graph['id'])) ?>">
                            <div>
                                <strong><?= h($graph['title']) ?></strong>
                                <span><?= h($graph['ticker_symbol']) ?></span>
                            </div>
                            <small><?= h((string) $graph['created_at']) ?></small>
                        </a>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </article>

        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Job History</p>
                    <h2>Recent queue activity for <?= h($query) ?></h2>
                </div>
            </div>
            <?php if ($jobHistory === []): ?>
                <p class="empty-state">No jobs have been queued for this ticker yet.</p>
            <?php else: ?>
                <div class="stack-list">
                    <?php foreach ($jobHistory as $job): ?>
                        <a class="list-card" href="<?= h(app_url('/view.php?job=' . (int) $job['id'])) ?>">
                            <div>
                                <strong>Job #<?= h((string) $job['id']) ?></strong>
                                <span><?= h($job['status']) ?></span>
                            </div>
                            <small><?= h((string) $job['created_at']) ?></small>
                        </a>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </article>
    </section>
<?php else: ?>
    <section class="content-grid two-up">
        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Browse</p>
                    <h2>Recently imported graphs</h2>
                </div>
            </div>
            <?php if ($recentGraphs === []): ?>
                <p class="empty-state">No graphs are stored yet. Queue your first run from the search box above.</p>
            <?php else: ?>
                <div class="stack-list">
                    <?php foreach ($recentGraphs as $graph): ?>
                        <a class="list-card" href="<?= h(app_url('/view.php?graph=' . (int) $graph['id'])) ?>">
                            <div>
                                <strong><?= h($graph['ticker_symbol']) ?></strong>
                                <span><?= h($graph['title']) ?></span>
                            </div>
                            <small><?= h((string) $graph['created_at']) ?></small>
                        </a>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </article>

        <article class="panel">
            <div class="panel-heading">
                <div>
                    <p class="eyebrow">Queue</p>
                    <h2>Current job list</h2>
                </div>
            </div>
            <?php if ($recentJobs === []): ?>
                <p class="empty-state">Nothing is in the queue yet.</p>
            <?php else: ?>
                <div class="stack-list">
                    <?php foreach ($recentJobs as $job): ?>
                        <a class="list-card" href="<?= h(app_url('/view.php?job=' . (int) $job['id'])) ?>">
                            <div>
                                <strong><?= h($job['ticker_symbol']) ?></strong>
                                <span><?= h($job['status']) ?></span>
                            </div>
                            <small><?= h((string) $job['created_at']) ?></small>
                        </a>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </article>
    </section>
<?php endif; ?>

<?php render_layout_end(); ?>

