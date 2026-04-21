<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: ' . app_url('/search.php'));
    exit;
}

$ticker = normalise_ticker((string) ($_POST['ticker'] ?? ''));
if ($ticker === '') {
    set_flash('Enter a valid ticker before queueing a forecast.', 'danger');
    header('Location: ' . app_url('/search.php'));
    exit;
}

$user = current_user();
$userId = isset($user['id']) && is_int($user['id']) ? $user['id'] : null;
$jobId = enqueue_prediction_job(db(), $userId, $ticker);
spawn_queue_worker();

set_flash(
    'Forecast request for ' . $ticker . ' added to the queue. If another run is active, this one will wait its turn.',
    'success'
);

header('Location: ' . app_url('/view.php?job=' . $jobId));
exit;

