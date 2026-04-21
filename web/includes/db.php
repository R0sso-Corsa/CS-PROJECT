<?php

require_once __DIR__ . '/config.php';

function db_connection_error_message($exception)
{
    return 'Database connection failed: ' . $exception->getMessage();
}

function db()
{
    static $pdo = null;

    if ($pdo instanceof PDO) {
        return $pdo;
    }

    $dsn = sprintf(
        'mysql:host=%s;port=%d;dbname=%s;charset=utf8mb4',
        DB_HOST,
        DB_PORT,
        DB_NAME
    );

    $pdo = new PDO(
        $dsn,
        DB_USER,
        DB_PASS,
        [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]
    );

    return $pdo;
}

function db_optional(&$error = null)
{
    try {
        return db();
    } catch (Exception $exception) {
        $error = db_connection_error_message($exception);
        return null;
    }
}
