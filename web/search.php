<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

function resolve_search_focus($query, $matches)
{
    if ($query === '' || $matches === []) {
        return null;
    }

    $exactSymbol = normalise_ticker($query);
    foreach ($matches as $match) {
        if (isset($match['symbol']) && (string) $match['symbol'] === $exactSymbol) {
            return $exactSymbol;
        }
    }

    if (count($matches) === 1 && isset($matches[0]['symbol'])) {
        return (string) $matches[0]['symbol'];
    }

    return null;
}

function can_queue_symbol_from_query($query)
{
    $query = normalise_search_query($query);
    if ($query === '') {
        return false;
    }

    $symbol = normalise_ticker($query);
    if ($symbol === '') {
        return false;
    }

    return preg_match('/^[A-Z0-9.\-=]{1,20}$/', $symbol) === 1
        && (preg_match('/[.\-=]/', $symbol) === 1 || strlen($symbol) <= 4);
}

$dbError = null;
$pdo = db_optional($dbError);
$rawQuery = isset($_GET['ticker']) ? (string) $_GET['ticker'] : '';
$query = normalise_search_query($rawQuery);
$matches = [];
$existingGraphs = [];
$jobHistory = [];
$recentGraphs = [];
$recentJobs = [];
$selectedTicker = null;
$queueTicker = null;
$yfinanceMatches = [];
$yfinanceError = null;

if ($query !== '') {
    log_yfinance_search_marker('search page starting yfinance lookup', ['query' => $query]);
    $yfinanceMatches = search_yfinance_tickers($query, 10, $yfinanceError);
    log_yfinance_search_marker('search page finished yfinance lookup', [
        'query' => $query,
        'count' => count($yfinanceMatches),
        'error' => $yfinanceError,
    ]);
}

if ($pdo instanceof PDO) {
    $matches = $query !== '' ? search_ticker_history($pdo, $query) : [];
    $selectedTicker = resolve_search_focus($query, $matches);
    if ($selectedTicker !== null) {
        $existingGraphs = fetch_graphs_for_ticker($pdo, $selectedTicker);
        $jobHistory = fetch_jobs_for_ticker($pdo, $selectedTicker);
        $queueTicker = $selectedTicker;
    } elseif (can_queue_symbol_from_query($query)) {
        $queueTicker = normalise_ticker($query);
    }
    $recentGraphs = $query === '' ? fetch_recent_graphs($pdo, 10) : [];
    $recentJobs = fetch_recent_jobs($pdo, 8);
}

render_layout_start('Search', 'search');
?>

<h1>Search ticker history</h1>
<p>Search by exact ticker, stored website history, or Yahoo Finance keyword results, then open a matching ticker or queue a new graph.</p>

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
                Ticker or keyword<br>
                <input
                    type="text"
                    name="ticker"
                    value="<?= h($query) ?>"
                    placeholder="BTC-USD, bitcoin, tesla, AAPL"
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
        <h2>Matching tickers for "<?= h($query) ?>"</h2>
        <?php if ($matches === []): ?>
            <p>No stored tickers matched this search.</p>
            <?php if ($queueTicker !== null && $pdo instanceof PDO): ?>
                <p>You can still queue this as a new exact ticker symbol if that is what you meant.</p>
            <?php else: ?>
                <p>Try an exact symbol such as AAPL or BTC-USD, or use a different keyword already present in saved graphs and job history.</p>
            <?php endif; ?>
        <?php else: ?>
            <ul>
                <?php foreach ($matches as $match): ?>
                    <li>
                        <a href="<?= h(app_url('/search.php?ticker=' . urlencode((string) $match['symbol']))) ?>">
                            <?= h((string) $match['symbol']) ?>
                        </a>
                        -
                        <?= h((string) $match['display_name']) ?>
                        -
                        <?= h((string) $match['graph_count']) ?> saved graphs
                        -
                        <?= h((string) $match['job_count']) ?> jobs
                        -
                        latest graph:
                        <?= h(isset($match['latest_graph_at']) && $match['latest_graph_at'] !== null ? (string) $match['latest_graph_at'] : 'none yet') ?>
                    </li>
                <?php endforeach; ?>
            </ul>
            <?php if ($selectedTicker === null): ?>
                <p>Choose one of the matching ticker links above to open its stored graphs and job history.</p>
            <?php endif; ?>
        <?php endif; ?>
    </section>

    <section>
        <h2>Yahoo Finance results for "<?= h($query) ?>"</h2>
        <?php if ($yfinanceError !== null): ?>
            <p><strong>Yahoo Finance lookup failed:</strong> <?= h($yfinanceError) ?></p>
        <?php elseif ($yfinanceMatches === []): ?>
            <p>No Yahoo Finance ticker matches were found.</p>
        <?php else: ?>
            <ul>
                <?php foreach ($yfinanceMatches as $result): ?>
                    <li>
                        <a href="<?= h(app_url('/search.php?ticker=' . urlencode((string) $result['symbol']))) ?>">
                            <?= h((string) $result['symbol']) ?>
                        </a>
                        -
                        <?= h(isset($result['name']) ? (string) $result['name'] : '') ?>
                        <?php if (!empty($result['exchange'])): ?>
                            -
                            <?= h((string) $result['exchange']) ?>
                        <?php endif; ?>
                        <?php if (!empty($result['type'])): ?>
                            -
                            <?= h((string) $result['type']) ?>
                        <?php endif; ?>
                    </li>
                <?php endforeach; ?>
            </ul>
            <p>Select a Yahoo Finance ticker above to search it directly and queue it for a forecast.</p>
        <?php endif; ?>
    </section>

    <?php if ($queueTicker !== null): ?>
    <section>
        <h2>Queue a fresh forecast for <?= h($queueTicker) ?></h2>
        <p>Submitting a new request adds the ticker to the queue if another training run is already active.</p>
        <?php if ($pdo instanceof PDO): ?>
            <form method="post" action="<?= h(app_url('/request_prediction.php')) ?>">
                <input type="hidden" name="ticker" value="<?= h($queueTicker) ?>">
                <button type="submit">Create New Graph</button>
            </form>
        <?php else: ?>
                <p>Queueing is disabled until the database connection is working.</p>
        <?php endif; ?>
    </section>
    <?php endif; ?>

    <?php if ($selectedTicker !== null): ?>
    <section>
        <h2>Stored summary for <?= h($selectedTicker) ?></h2>
        <?php if ($matches === []): ?>
            <p>This ticker is not in the database yet.</p>
        <?php else: ?>
            <ul>
                <?php foreach ($matches as $match): ?>
                    <?php if ((string) $match['symbol'] !== $selectedTicker) { continue; } ?>
                    <li>
                        <?= h($match['symbol']) ?> -
                        <?= h((string) $match['graph_count']) ?> saved graphs -
                        <?= h(isset($match['latest_graph_at']) && $match['latest_graph_at'] !== null ? (string) $match['latest_graph_at'] : 'No graph imported yet') ?>
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
        <h2>Job history for <?= h($selectedTicker) ?></h2>
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
    <?php endif; ?>
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
