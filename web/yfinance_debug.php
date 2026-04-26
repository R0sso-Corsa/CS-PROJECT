<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$query = isset($_GET['ticker']) ? normalise_search_query((string) $_GET['ticker']) : 'tesla';
$script = WEB_ROOT . '/tools/search_yfinance.py';
$command = sprintf(
    '%s %s %s --limit %d',
    escapeshellarg(YFINANCE_SEARCH_PYTHON),
    escapeshellarg($script),
    escapeshellarg($query),
    5
);

$output = [];
$exitCode = 0;
$canExec = function_exists('exec');
$raw = '';

if ($canExec) {
    exec($command . ' 2>&1', $output, $exitCode);
    $raw = trim(implode("\n", $output));
}

$decoded = json_decode($raw, true);
if (!is_array($decoded)) {
    foreach ($output as $line) {
        $candidate = trim((string) $line);
        if ($candidate === '' || $candidate[0] !== '{') {
            continue;
        }

        $candidateDecoded = json_decode($candidate, true);
        if (is_array($candidateDecoded)) {
            $decoded = $candidateDecoded;
            break;
        }
    }
}

render_layout_start('Yahoo Finance Debug', 'search');
?>

<h1>Yahoo Finance debug</h1>
<p>This page runs the same yfinance helper that the search page uses.</p>

<form method="get">
    <p>
        <label>
            Search query<br>
            <input type="text" name="ticker" value="<?= h($query) ?>">
        </label>
        <button type="submit">Run Test</button>
    </p>
</form>

<section>
    <h2>Environment</h2>
    <dl>
        <dt>WEB_ROOT</dt>
        <dd><?= h(WEB_ROOT) ?></dd>
        <dt>Helper path</dt>
        <dd><?= h($script) ?></dd>
        <dt>Helper exists</dt>
        <dd><?= is_file($script) ? 'yes' : 'no' ?></dd>
        <dt>Python setting</dt>
        <dd><?= h(YFINANCE_SEARCH_PYTHON) ?></dd>
        <dt>exec available</dt>
        <dd><?= $canExec ? 'yes' : 'no' ?></dd>
        <dt>Log path</dt>
        <dd><?= h(LOG_ROOT . '/yfinance-search.log') ?></dd>
        <dt>Log folder exists</dt>
        <dd><?= is_dir(LOG_ROOT) ? 'yes' : 'no' ?></dd>
        <dt>Log folder writable</dt>
        <dd><?= is_writable(LOG_ROOT) ? 'yes' : 'no' ?></dd>
    </dl>
</section>

<section>
    <h2>Command</h2>
    <pre><?= h($command) ?></pre>
    <p>Exit code: <?= h((string) $exitCode) ?></p>
</section>

<section>
    <h2>Raw output</h2>
    <pre><?= h($raw !== '' ? $raw : '[empty]') ?></pre>
</section>

<section>
    <h2>Parsed output</h2>
    <?php if (is_array($decoded)): ?>
        <pre><?= h(json_encode($decoded, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES)) ?></pre>
    <?php else: ?>
        <p>Output could not be parsed as JSON.</p>
    <?php endif; ?>
</section>

<?php render_layout_end(); ?>
