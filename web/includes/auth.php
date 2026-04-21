<?php

require_once __DIR__ . '/db.php';

function ensure_session_started()
{
    if (session_status() !== PHP_SESSION_ACTIVE) {
        session_start();
    }
}

function current_user()
{
    ensure_session_started();
    return isset($_SESSION['user']) ? $_SESSION['user'] : null;
}

function set_flash($message, $type = 'info')
{
    ensure_session_started();
    $_SESSION['flash'] = [
        'message' => $message,
        'type' => $type,
    ];
}

function pull_flash()
{
    ensure_session_started();
    if (!isset($_SESSION['flash'])) {
        return null;
    }

    $flash = $_SESSION['flash'];
    unset($_SESSION['flash']);
    return $flash;
}

function establish_user_session($id, $username, $isDemo)
{
    $_SESSION['user'] = [
        'id' => $id,
        'username' => $username,
        'is_demo' => $isDemo,
    ];
}

function login_user($username, $password, &$error = null)
{
    ensure_session_started();

    if ($username === DEMO_USERNAME && $password === DEMO_PASSWORD) {
        establish_user_session(null, DEMO_USERNAME, true);
        return true;
    }

    try {
        $stmt = db()->prepare(
            'SELECT id, username, password_hash
             FROM app_users
             WHERE username = :username
             LIMIT 1'
        );
        $stmt->execute(['username' => $username]);
        $user = $stmt->fetch();
    } catch (Exception $exception) {
        $error = 'The database login is not available right now. If the database is not configured yet, use the demo login: demo / demo123.';
        return false;
    }

    if (is_array($user) && password_verify($password, (string) $user['password_hash'])) {
        establish_user_session((int) $user['id'], (string) $user['username'], false);
        return true;
    }

    $error = 'Those credentials were not accepted.';
    return false;
}

function create_user($username, $password, $confirmPassword, &$error = null)
{
    ensure_session_started();

    $username = trim($username);

    if ($username === '') {
        $error = 'Enter a username.';
        return false;
    }

    if (!preg_match('/^[A-Za-z0-9_.-]{3,40}$/', $username)) {
        $error = 'Use 3 to 40 characters made of letters, numbers, dots, dashes, or underscores.';
        return false;
    }

    if (strcasecmp($username, DEMO_USERNAME) === 0) {
        $error = 'That username is reserved. Please choose another one.';
        return false;
    }

    if (strlen($password) < 8) {
        $error = 'Use a password with at least 8 characters.';
        return false;
    }

    if ($password !== $confirmPassword) {
        $error = 'The passwords do not match.';
        return false;
    }

    try {
        $check = db()->prepare(
            'SELECT id
             FROM app_users
             WHERE username = :username
             LIMIT 1'
        );
        $check->execute(['username' => $username]);
        if ($check->fetch() !== false) {
            $error = 'That username already exists.';
            return false;
        }

        $insert = db()->prepare(
            'INSERT INTO app_users (username, password_hash)
             VALUES (:username, :password_hash)'
        );
        $insert->execute([
            'username' => $username,
            'password_hash' => password_hash($password, PASSWORD_DEFAULT),
        ]);

        establish_user_session((int) db()->lastInsertId(), $username, false);
        return true;
    } catch (Exception $exception) {
        $error = 'Account creation is not available right now. Check that the database is configured and writable.';
        return false;
    }
}

function logout_user()
{
    ensure_session_started();
    $_SESSION = [];
    session_destroy();
}

function require_login()
{
    if (current_user() !== null) {
        return;
    }

    set_flash('Log in to request a forecast or browse saved graphs.', 'warning');
    redirect_to('/login.php');
}
