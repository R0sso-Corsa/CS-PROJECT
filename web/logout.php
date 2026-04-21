<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/bootstrap.php';

logout_user();
ensure_session_started();
set_flash('You have been logged out.', 'success');
redirect_to('/login.php');
