<?php

require_once __DIR__ . '/includes/bootstrap.php';

if (current_user() !== null) {
    redirect_to('/index.php');
}

$error = null;
$username = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = isset($_POST['username']) ? trim((string) $_POST['username']) : '';
    $password = isset($_POST['password']) ? (string) $_POST['password'] : '';
    $confirmPassword = isset($_POST['confirm_password']) ? (string) $_POST['confirm_password'] : '';

    if (create_user($username, $password, $confirmPassword, $error)) {
        set_flash('Your account has been created and you are now signed in.', 'success');
        redirect_to('/index.php');
    }
}

render_layout_start('Create Account', 'register');
?>

<h1>Create account</h1>
<p>This creates a real user in the site database and signs you in automatically after success.</p>

<?php if ($error !== null): ?>
    <section>
        <p><strong>Error:</strong> <?= h($error) ?></p>
    </section>
<?php endif; ?>

<form method="post">
    <p>
        <label>
            Username<br>
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
    </p>
    <p>
        <label>
            Password<br>
            <input type="password" name="password" placeholder="At least 8 characters" minlength="8" required>
        </label>
    </p>
    <p>
        <label>
            Confirm password<br>
            <input type="password" name="confirm_password" placeholder="Repeat your password" minlength="8" required>
        </label>
    </p>
    <p>
        <button type="submit">Create Account</button>
    </p>
</form>

<p><a href="<?= h(app_url('/login.php')) ?>">Back to Log In</a></p>

<?php render_layout_end(); ?>
