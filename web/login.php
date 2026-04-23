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

<h1>Log in</h1>
<p>Use a real database account if one exists, or use the demo fallback while the rest of the stack is still being configured.</p>

<?php if ($error !== null): ?>
    <section>
        <p><strong>Error:</strong> <?= h($error) ?></p>
    </section>
<?php endif; ?>

<form method="post">
    <p>
        <label>
            Username<br>
            <input type="text" name="username" placeholder="demo" required>
        </label>
    </p>
    <p>
        <label>
            Password<br>
            <input type="password" name="password" placeholder="demo123" required>
        </label>
    </p>
    <p>
        <button type="submit">Log In</button>
        <a href="<?= h(app_url('/register.php')) ?>">Create Account</a>
    </p>
</form>

<p><strong>Demo fallback:</strong> <?= h(DEMO_USERNAME) ?> / <?= h(DEMO_PASSWORD) ?></p>

<?php render_layout_end(); ?>
