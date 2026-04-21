<?php

require_once __DIR__ . '/includes/bootstrap.php';

if (current_user() !== null) {
    redirect_to('/index.php');
}

$error = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = isset($_POST['username']) ? trim((string) $_POST['username']) : '';
    $password = isset($_POST['password']) ? (string) $_POST['password'] : '';

    if (login_user($username, $password, $error)) {
        set_flash('You are now signed in.', 'success');
        redirect_to('/index.php');
    }
}

render_layout_start('Log In', 'login');
?>

<section class="auth-shell">
    <article class="auth-panel">
        <p class="eyebrow">Account Access</p>
        <h1>Log in to search, queue, and review forecasts.</h1>
        <p class="support-copy">
            Use a real database account if you have seeded one, or use the demo fallback while you are still wiring the rest of the stack.
        </p>

        <?php if ($error !== null): ?>
            <div class="flash flash-danger"><?= h($error) ?></div>
        <?php endif; ?>

        <form class="auth-form" method="post">
            <label>
                <span>Username</span>
                <input type="text" name="username" placeholder="demo" required>
            </label>
            <label>
                <span>Password</span>
                <input type="password" name="password" placeholder="demo123" required>
            </label>
            <div class="auth-links auth-links--inline">
                <button class="button button-primary" type="submit">Log In</button>
                <a class="button button-secondary" href="<?= h(app_url('/register.php')) ?>">Create Account</a>
            </div>
        </form>

        <div class="auth-hint">
            <strong>Demo fallback:</strong> <?= h(DEMO_USERNAME) ?> / <?= h(DEMO_PASSWORD) ?>
        </div>
    </article>
</section>

<?php render_layout_end(); ?>
