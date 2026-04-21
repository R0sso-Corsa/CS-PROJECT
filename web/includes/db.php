<?php
declare(strict_types=1);

require_once __DIR__ . '/config.php';

function db_connection_error_message(Throwable $exception): string
{
    return 'Database connection failed: ' . $exception->getMessage();
}

function db(): PDO
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

function db_optional(?string &$error = null): ?PDO
{
    try {
        return db();
    } catch (Throwable $exception) {
        $error = db_connection_error_message($exception);
        return null;
    }
}
