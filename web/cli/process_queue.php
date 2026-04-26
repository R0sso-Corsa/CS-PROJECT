<?php

require_once dirname(__DIR__) . '/includes/config.php';
require_once dirname(__DIR__) . '/includes/db.php';
require_once dirname(__DIR__) . '/includes/jobs.php';

try {
    $result = process_prediction_queue(db());
    echo 'Queue worker result: ' . json_encode($result, JSON_UNESCAPED_SLASHES) . PHP_EOL;
    exit(isset($result['failed']) && (int) $result['failed'] > 0 ? 1 : 0);
} catch (Throwable $exception) {
    fwrite(STDERR, 'Queue worker crashed: ' . $exception->getMessage() . PHP_EOL);
    exit(1);
}
