<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/bootstrap.php';

$stats = fetch_dashboard_stats(db());
$recentJobs = fetch_recent_jobs(db(), 5);
$recentGraphs = fetch_recent_graphs(db(), 4);

render_layout_start('Home', 'home');
?>

<section class="hero-panel">
    <div>
        <p class="eyebrow">Forecast Queue + Graph Archive</p>
        <h1>Queue training requests, save finished graphs, and keep each ticker’s history in one place.</h1>
        <p class="hero-copy">
            This starter site is designed for your Linux XAMPP deployment. PHP handles the login flow, stores graph records in MySQL,
            and queues forecast jobs so only one remote training run is active at a time.
        </p>
        <div class="hero-actions">
            <a class="button button-primary" href="<?= h(app_url('/search.php')) ?>">Open Search Page</a>
            <a class="button button-secondary" href="<?= h(app_url('/login.php')) ?>">Log In</a>
            <a class="button button-secondary" href="<?= h(app_url('/register.php')) ?>">Create Account</a>
        </div>
    </div>
    <div class="hero-grid">
        <article class="metric-card">
            <span>Queued Jobs</span>
            <strong><?= h((string) $stats['queued_jobs']) ?></strong>
        </article>
        <article class="metric-card">
            <span>Running Jobs</span>
            <strong><?= h((string) $stats['running_jobs']) ?></strong>
        </article>
        <article class="metric-card">
            <span>Saved Graphs</span>
            <strong><?= h((string) $stats['saved_graphs']) ?></strong>
        </article>
        <article class="metric-card">
            <span>Tracked Tickers</span>
            <strong><?= h((string) $stats['tracked_tickers']) ?></strong>
        </article>
    </div>
</section>

<section class="content-grid two-up">
    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Workflow</p>
                <h2>How the site behaves</h2>
            </div>
        </div>
        <ol class="step-list">
            <li>Users log in and search for a ticker.</li>
            <li>They either open an older saved graph or request a new forecast.</li>
            <li>New forecast requests are written into the `prediction_jobs` queue table.</li>
            <li>A background PHP worker sends a single SSH command to this PC to start training.</li>
            <li>When the run finishes, plots are imported back into the site and stored in the database-backed archive.</li>
        </ol>
    </article>

    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Demo Access</p>
                <h2>Quick local login</h2>
            </div>
        </div>
        <p class="support-copy">
            If you have not created database users yet, you can still test the interface with the demo fallback account below.
        </p>
        <dl class="demo-credentials">
            <div>
                <dt>Username</dt>
                <dd><?= h(DEMO_USERNAME) ?></dd>
            </div>
            <div>
                <dt>Password</dt>
                <dd><?= h(DEMO_PASSWORD) ?></dd>
            </div>
        </dl>
    </article>
</section>

<section class="content-grid two-up">
    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Recent Jobs</p>
                <h2>Queue activity</h2>
            </div>
        </div>
        <?php if ($recentJobs === []): ?>
            <p class="empty-state">No jobs have been queued yet.</p>
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

    <article class="panel">
        <div class="panel-heading">
            <div>
                <p class="eyebrow">Recent Graphs</p>
                <h2>Latest saved views</h2>
            </div>
        </div>
        <?php if ($recentGraphs === []): ?>
            <p class="empty-state">No graph images have been imported yet.</p>
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
</section>

<?php render_layout_end(); ?>
