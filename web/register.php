<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/bootstrap.php';

if (current_user() !== null) {
    header('Location: ' . app_url('/search.php'));
    exit;
}

$error = null;
$username = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim((string) ($_POST['username'] ?? ''));
    $password = (string) ($_POST['password'] ?? '');
    $confirmPassword = (string) ($_POST['confirm_password'] ?? '');

    if (create_user($username, $password, $confirmPassword, $error)) {
        set_flash('Your account has been created and you are now signed in.', 'success');
        header('Location: ' . app_url('/search.php'));
        exit;
    }
}

render_layout_start('Create Account', 'register');
?>

<section class="auth-shell">
    <article class="auth-panel">
        <p class="eyebrow">New Account</p>
        <h1>Create an account to queue and review forecasts.</h1>
        <p class="support-copy">
            This creates a real user in the site database. Once the account is created, you will be signed in automatically.
        </p>

        <?php if ($error !== null): ?>
            <div class="flash flash-danger"><?= h($error) ?></div>
        <?php endif; ?>

        <form class="auth-form" method="post">
            <label>
                <span>Username</span>
                <input
                    type="text"
                    name="username"
                    value="<?= h($username) ?>"
                    placeholder="newuser"
                    minlength="3"
                    maxlength="40"
                    required
                >
            </label>
            <label>
                <span>Password</span>
                <input type="password" name="password" placeholder="At least 8 characters" minlength="8" required>
            </label>
            <label>
                <span>Confirm Password</span>
                <input type="password" name="confirm_password" placeholder="Repeat your password" minlength="8" required>
            </label>
            <button class="button button-primary" type="submit">Create Account</button>
        </form>

        <div class="auth-links">
            <a class="button button-secondary" href="<?= h(app_url('/login.php')) ?>">Back to Log In</a>
        </div>
    </article>
</section>

<?php render_layout_end(); ?>
