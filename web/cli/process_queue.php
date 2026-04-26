<?php

error_reporting(E_ALL);
ini_set('display_errors', '1');

echo 'Queue worker booting from ' . __FILE__ . PHP_EOL;

try {
    require_once dirname(__DIR__) . '/includes/config.php';
    require_once dirname(__DIR__) . '/includes/db.php';
    require_once dirname(__DIR__) . '/includes/jobs.php';

    $result = process_prediction_queue(db());
    echo 'Queue worker result: ' . json_encode($result, JSON_UNESCAPED_SLASHES) . PHP_EOL;
    exit(isset($result['failed']) && (int) $result['failed'] > 0 ? 1 : 0);
} catch (Throwable $exception) {
    fwrite(STDERR, 'Queue worker crashed: ' . $exception->getMessage() . PHP_EOL);
    exit(1);
}
