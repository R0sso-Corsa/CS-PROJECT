<?php
declare(strict_types=1);

require_once __DIR__ . '/db.php';

function ensure_session_started(): void
{
    if (session_status() !== PHP_SESSION_ACTIVE) {
        session_start();
    }
}

function current_user(): ?array
{
    ensure_session_started();
    return $_SESSION['user'] ?? null;
}

function set_flash(string $message, string $type = 'info'): void
{
    ensure_session_started();
    $_SESSION['flash'] = [
        'message' => $message,
        'type' => $type,
    ];
}

function pull_flash(): ?array
{
    ensure_session_started();
    if (!isset($_SESSION['flash'])) {
        return null;
    }

    $flash = $_SESSION['flash'];
    unset($_SESSION['flash']);
    return $flash;
}

function login_user(string $username, string $password): bool
{
    ensure_session_started();

    $stmt = db()->prepare(
        'SELECT id, username, password_hash
         FROM app_users
         WHERE username = :username
         LIMIT 1'
    );
    $stmt->execute(['username' => $username]);
    $user = $stmt->fetch();

    if (is_array($user) && password_verify($password, (string) $user['password_hash'])) {
        $_SESSION['user'] = [
            'id' => (int) $user['id'],
            'username' => (string) $user['username'],
            'is_demo' => false,
        ];
        return true;
    }

    if ($username === DEMO_USERNAME && $password === DEMO_PASSWORD) {
        $_SESSION['user'] = [
            'id' => null,
            'username' => DEMO_USERNAME,
            'is_demo' => true,
        ];
        return true;
    }

    return false;
}

function logout_user(): void
{
    ensure_session_started();
    $_SESSION = [];
    session_destroy();
}

function require_login(): void
{
    if (current_user() !== null) {
        return;
    }

    set_flash('Log in to request a forecast or browse saved graphs.', 'warning');
    header('Location: ' . app_url('/login.php'));
    exit;
}

