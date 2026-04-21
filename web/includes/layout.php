<?php
declare(strict_types=1);

require_once __DIR__ . '/auth.php';

function app_url(string $path = ''): string
{
    $base = rtrim(APP_BASE_URL, '/');
    $tail = '/' . ltrim($path, '/');
    if ($base === '') {
        return $tail;
    }

    return $base . $tail;
}

function h(?string $value): string
{
    return htmlspecialchars((string) $value, ENT_QUOTES, 'UTF-8');
}

function render_layout_start(string $title, string $active = ''): void
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

function render_layout_end(): void
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

