<?php

require_once __DIR__ . '/auth.php';

function app_base_path()
{
    if (APP_BASE_URL !== '') {
        return '/' . trim(APP_BASE_URL, '/');
    }

    $scriptName = isset($_SERVER['SCRIPT_NAME']) ? (string) $_SERVER['SCRIPT_NAME'] : '';
    if ($scriptName === '') {
        return '';
    }

    $dir = str_replace('\\', '/', dirname($scriptName));
    if ($dir === '/' || $dir === '.') {
        return '';
    }

    return '/' . trim($dir, '/');
}

function app_url($path = '')
{
    $base = rtrim(app_base_path(), '/');
    $tail = '/' . ltrim($path, '/');
    if ($base === '') {
        return $tail;
    }

    return $base . $tail;
}

function h($value)
{
    return htmlspecialchars((string) $value, ENT_QUOTES, 'UTF-8');
}

function display_job_status($status)
{
    $status = strtolower(trim((string) $status));
    if ($status === 'completed') {
        return 'processed';
    }
    return $status;
}

function redirect_to($path, $status = 302)
{
    $target = app_url($path);

    if (!headers_sent()) {
        header('Cache-Control: no-store, no-cache, must-revalidate, max-age=0');
        header('Pragma: no-cache');
        header('Location: ' . $target, true, $status);
    }

    $escaped = h($target);
    echo '<!DOCTYPE html><html lang="en"><head>';
    echo '<meta charset="utf-8">';
    echo '<meta http-equiv="refresh" content="0;url=' . $escaped . '">';
    echo '<title>Redirecting...</title>';
    echo '<script>window.location.replace(' . json_encode($target, JSON_UNESCAPED_SLASHES) . ');</script>';
    echo '</head><body>';
    echo '<p>Redirecting to <a href="' . $escaped . '">' . $escaped . '</a>...</p>';
    echo '</body></html>';
    exit;
}

function render_layout_start($title, $active = '')
{
    ensure_session_started();
    $user = current_user();
    $flash = pull_flash();
    ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title><?= h($title) ?> | <?= h(APP_NAME) ?></title>
    <link rel="stylesheet" href="<?= h(app_url('/stylesheet.css')) ?>">
    <script defer src="<?= h(app_url('/scripts.js')) ?>"></script>
</head>
<body>
<header>
    <p><strong><?= h(APP_NAME) ?></strong></p>
    <p>Queued ticker forecasting website.</p>
    <nav>
        <a href="<?= h(app_url('/index.php')) ?>">Home</a>
        |
        <a href="<?= h(app_url('/search.php')) ?>">Search</a>
        |
        <a href="<?= h(app_url('/healthcheck.php')) ?>">Health Check</a>
        <?php if ($user !== null): ?>
            |
            <span>Signed in as <?= h($user['username']) ?></span>
            |
            <a href="<?= h(app_url('/logout.php')) ?>">Log Out</a>
        <?php else: ?>
            |
            <a href="<?= h(app_url('/login.php')) ?>">Log In</a>
            |
            <a href="<?= h(app_url('/register.php')) ?>">Create Account</a>
        <?php endif; ?>
    </nav>
    <hr>
</header>
<main>
    <?php if ($flash !== null): ?>
        <section>
            <p><strong><?= h(ucfirst((string) $flash['type'])) ?>:</strong> <?= h($flash['message']) ?></p>
        </section>
    <?php endif; ?>
    <?php
}

function render_layout_end()
{
    ?>
</main>
<hr>
<footer>
    <p>Built for Linux XAMPP deployment with PHP queueing, SSH-triggered training, and database-backed graph history.</p>
</footer>
</body>
</html>
    <?php
}
