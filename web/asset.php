<?php

require_once __DIR__ . '/includes/bootstrap.php';

require_login();

$graphId = isset($_GET['graph']) ? (int) $_GET['graph'] : 0;
$kind = isset($_GET['kind']) ? (string) $_GET['kind'] : 'summary';
if (!in_array($kind, ['summary', 'detail', 'residuals'], true) || $graphId <= 0) {
    http_response_code(404);
    exit;
}

$stmt = db()->prepare(
    "SELECT mime_type, binary_data
     FROM saved_graph_assets
     WHERE graph_id = :graph_id AND asset_kind = :asset_kind
     LIMIT 1"
);
$stmt->execute([
    'graph_id' => $graphId,
    'asset_kind' => $kind,
]);
$asset = $stmt->fetch();

if (!is_array($asset)) {
    http_response_code(404);
    exit;
}

header('Content-Type: ' . (string) $asset['mime_type']);
echo $asset['binary_data'];
