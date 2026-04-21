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
<div class="page-shell">
    <header class="site-header">
        <a class="brand-mark" href="<?= h(app_url('/index.php')) ?>">
            <span class="brand-dot"></span>
            <span>
                <strong><?= h(APP_NAME) ?></strong>
                <small>Queued ticker forecasting</small>
            </span>
        </a>
        <nav class="site-nav">
            <a class="<?= $active === 'home' ? 'active' : '' ?>" href="<?= h(app_url('/index.php')) ?>">Home</a>
            <a class="<?= $active === 'search' ? 'active' : '' ?>" href="<?= h(app_url('/search.php')) ?>">Search</a>
            <?php if ($user !== null): ?>
                <span class="nav-user">Signed in as <?= h($user['username']) ?></span>
                <a href="<?= h(app_url('/logout.php')) ?>">Log Out</a>
            <?php else: ?>
                <a class="<?= $active === 'login' ? 'active' : '' ?>" href="<?= h(app_url('/login.php')) ?>">Log In</a>
                <a class="<?= $active === 'register' ? 'active' : '' ?>" href="<?= h(app_url('/register.php')) ?>">Create Account</a>
            <?php endif; ?>
        </nav>
    </header>
    <main class="page-main">
        <?php if ($flash !== null): ?>
            <div class="flash flash-<?= h($flash['type']) ?>">
                <?= h($flash['message']) ?>
            </div>
        <?php endif; ?>
    <?php
}

function render_layout_end()
{
    ?>
    </main>
    <footer class="site-footer">
        <p>Built for a Linux XAMPP deployment with PHP queueing, SSH-triggered training, and database-backed graph history.</p>
    </footer>
</div>
</body>
</html>
    <?php
}
