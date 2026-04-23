# Website Build Notes: What I Built, What Went Wrong, and How I Fixed It

## Purpose of this document

This document explains the website and server work I added to the forecasting project. It describes what I built, why I built it that way, which problems appeared during testing and deployment, and what I did to fix them. I have written it in a simpler A-level style so that the technical ideas are still accurate but easier to understand.

## Main aim of the website

The website was not meant to be just a set of pages that looked good. It had to do real jobs for the project. It needed to let a user sign in, search for a ticker, look at older saved graphs, request a new forecast, and link the Linux XAMPP website to the Windows PC that runs the training scripts. Because of that, the website had to be built around the real system rather than around appearance alone.

## Original design choices

### 1. I organised the website around the real user journey

I split the website into several pages because each page had a clear purpose. `index.php` was used as the home page, `login.php` was for signing in, `register.php` was for creating an account, `search.php` was for entering and checking ticker symbols, and `view.php` was for seeing graphs or job progress. This made the structure easier to understand and easier to test.

### 2. I moved shared code into `includes`

I did not want every page to repeat the same code. So I placed shared logic such as database access, authentication, layout handling, and queue functions into `web/includes/`. This made the code more organised and made it easier to fix bugs in one place instead of changing the same logic in many different files.

### 3. I used a queue instead of trying to train directly from the page

The forecasting scripts can take a long time to run, so it would have been a bad idea to start training inside a normal browser request. If I had done that, the page could have timed out or looked broken to the user. Instead, the site writes a request into the `prediction_jobs` table and a separate worker handles the job afterwards.

### 4. I only allowed one active training job at a time

The project needed to avoid running many heavy jobs at once. The Windows training PC is the main limit in the system, so I chose to let only one job run at a time. This made the system simpler, safer, and easier to explain. It also matched the project requirement that new jobs should wait if training is already happening.

### 5. I stored ticker history in a relational database structure

Instead of making a whole new database for every ticker, I used a normal relational structure. I used `tickers`, `prediction_jobs`, `saved_graphs`, and `saved_graph_assets`. This still gives each ticker its own history, but it is much easier to manage and search.

### 6. I added a helper script on the Windows side

I added `web/tools/run_remote_prediction_job.py` so the Linux website would not have to guess where the output files were saved. The helper script runs the training, creates the graph images, and returns a JSON file listing the important output paths. This made the website-to-training link much clearer.

## Main website features I added

### 1. I replaced the old static prototype with a real PHP login system

The old `web` folder had a very simple prototype with basic HTML and JavaScript. It was useful as early evidence, but it was not strong enough to become the real site. I replaced it with a PHP version that uses sessions, protected pages, flash messages, and proper login logic.

### 2. I added account registration

I created `register.php` and added the logic needed to store new users in the database. This included password checks and password hashing. That meant the site could move beyond just using one fallback demo account.

### 3. I improved redirects and path handling

When I tested the website in different deployment setups, I found that links and redirects could break if the site was not in exactly the folder I expected. I fixed this by improving the helper that builds URLs. It can now work out the base path more safely, which made the site more reliable when deployed.

### 4. I added a health check page

I created `healthcheck.php` so I could quickly test whether the deployed site was working properly. It checks whether PHP is running, whether sessions work, whether storage folders are writable, and whether the database can be reached. This became very useful because many of the early website problems were actually server setup problems.

### 5. I added frontend debug logging

At one point it was not always clear whether buttons and forms were working properly in the browser. To make debugging easier, I added JavaScript logging so button presses and form submissions could be tracked. This helped me work out whether a problem was happening in the browser or later in the server logic.

## Deployment and server work

### 1. The website had to be prepared for Linux XAMPP

The site was not only for local Windows testing. It had to run on Linux XAMPP. This meant I had to think about Linux paths, document root locations, and how the site would behave inside `/opt/lampp/htdocs`.

### 2. Older XAMPP and PHP setups caused extra problems

The local machine did not have `php` on the PATH, so I could not fully lint the PHP files locally from the command line. That meant some checks had to be done by careful reading first and then tested properly on Linux. Later I also made the code more compatible with older XAMPP and PHP setups.

### 3. File permissions caused many of the deployment errors

A lot of the early website errors were actually caused by Linux file permissions. Apache could not always read the site files, and the application could not always write to storage or logs. This taught me that a website can look broken even when the code is fine, simply because the server does not have permission to use the files it needs.

### 4. Storage folders and log files had to be created properly

The queue worker expected certain folders and log files to exist. If they were missing, the queue system could fail even though the pages themselves loaded. I had to make sure the folders for logs and imported graph files existed and were writable.

### 5. The site had to use the correct CLI PHP binary

The queue worker runs from the command line, not just through Apache. On the Linux server the correct PHP binary was `/opt/lampp/bin/php`, not plain `php`. This small detail mattered because the worker would fail if it tried to use the wrong executable.

## SSH and Windows integration

### 1. The hardest part was linking Linux to Windows

The most difficult part of the system was not the HTML layout. It was the connection between the Linux XAMPP server and the Windows PC that runs the forecasting scripts. The Linux server needed to be able to start the training remotely through SSH without asking for manual input.

### 2. SSH setup took several steps

I had to identify the right Windows username, generate the correct SSH key on Linux, place the public key on Windows, and check that the Windows OpenSSH server was actually running and listening on port `22`. There was also confusion between a Windows Hello PIN, a Windows account password, and an SSH passphrase, so I had to separate those ideas clearly.

### 3. Windows administrator key handling was a hidden problem

One of the less obvious problems was that the Windows user was part of the `Administrators` group. Because of that, OpenSSH did not use the normal `authorized_keys` file in the usual user profile location. Instead, it used the administrator key file in `ProgramData`. Until I understood that, the correct key was still being rejected.

### 4. The website config had to match the values that actually worked

Even after I got manual SSH working, the website could still fail if its config file used the wrong host, user, key path, Python path, or repo path. So the PHP config had to be updated so it matched the exact values that already worked in manual tests.

## Queue-worker design and later fixes

### 1. The worker could connect over SSH but still fail to run the job

Getting SSH to connect was only one step. After that, the worker still had to run the correct command on Windows, wait for the helper script to finish, and then import the results. This meant that the queue could still fail even after the network connection itself was working.

### 2. Failed jobs did not restart by themselves

When a job failed, it stayed in the `failed` state. This was actually a safe design choice, but during debugging it meant I had to either re-queue jobs or reset their status manually before I could test the fix properly.

### 3. A major bug came from Windows command quoting

One of the clearest bugs was caused by the way the Linux worker built the remote Windows command. It used Unix-style quoting, but Windows PowerShell and the Windows shell do not read that format the same way. This caused a broken command to reach Windows. I fixed this by changing the worker so it launched the helper through PowerShell with Windows-safe quoting.

## Public access and exposure

### 1. Making the site public became a separate problem

Once the site mostly worked inside the local network, the next issue was how other people would reach it. That meant I had to think about safe public exposure, not just internal functionality.

### 2. I compared three main options

I looked at Cloudflare Tunnel, Tailscale Funnel, and direct router port forwarding. Cloudflare Tunnel seemed the safest long-term public option, Tailscale Funnel seemed the quickest secure sharing option, and direct port forwarding was the most traditional but also the riskiest for a home setup.

### 3. Tailscale Funnel became the main route I investigated

The Tailscale work showed that simply turning Funnel on is not enough. It still needs a real local service to publish. In this case that meant linking it properly to the XAMPP website on `http://127.0.0.1:80`.

## Detailed errors and how I fixed them

### 1. Login looked like it was not working

At one stage, clicking `Log In` seemed to do nothing. The real cause was that the login flow tried to contact the database before checking the fallback demo account. If the database was not ready, even the demo login could fail. I fixed this by changing the login flow so the demo account could be checked even when the database was unavailable. That gave me a safe way to test the site while the database was still being set up.

### 2. Redirects and links were wrong in some deployments

After login and page changes, some redirects did not lead to the right place. This happened because the site was assuming a simpler path structure than the deployed Linux XAMPP setup really used. I fixed this by improving the URL helper so it could detect the correct base path and build links more safely. This made page changes much more reliable.

### 3. I saw plain internal server errors

There were several times when pages showed a basic internal server error without much visible information. This was frustrating because the same error page could be caused by different problems. I learned that I had to check the logs every time instead of guessing. In different cases, the real cause was a file permission problem, a broken config file, or another server-side fatal error.

### 4. Apache could not read `includes/bootstrap.php`

One important error was that Apache could not open `/opt/lampp/htdocs/includes/bootstrap.php`. This meant the code itself might have been fine, but the server did not have permission to read it. I fixed this by correcting file and directory permissions under the deployed web root so Apache could read the includes properly.

### 5. Storage and log folders were missing or not writable

The queue system expected folders such as `storage/logs` to exist and be writable. When they were missing or locked down, the queue worker could fail even though the website pages still loaded. I fixed this by creating the needed folders and log files and making sure the deployment user had permission to write to them.

### 6. An empty worker log did not always mean failure

At one point I thought the worker was not doing anything because the log file stayed empty. Later I realised that the worker does not always log normal progress. So instead of relying only on the log, I started checking the CLI output, the PHP error log, and the database job table. That gave a much better picture of what the worker was really doing.

### 7. `config.php` caused more than one runtime fatal error

The deployment config file caused several different problems. One possible mistake was adding a second set of `REMOTE_*` constants instead of replacing the old ones. Another logged issue was `Undefined constant "DIR"`, which showed the config file had been edited incorrectly. I fixed these problems by treating the config file more carefully, making sure each constant only appeared once, and checking that the syntax matched the working values from manual tests.

### 8. XAMPP and PHP compatibility made some errors harder to understand

Because the Linux XAMPP setup was not the same as a modern local PHP environment, some problems could look like code bugs when they were really environment differences. I responded by making the PHP code more conservative and testing it on the real Linux deployment instead of trusting assumptions from the local machine.

### 9. Windows SSH setup failed before key auth was even reached

Some early SSH problems had nothing to do with the key itself. The Windows `.ssh` folder was not always present, and installing or enabling OpenSSH sometimes needed an administrator PowerShell session. In some cases the connection just hung because the SSH server was not fully ready. I fixed these issues by creating the correct paths, using an elevated shell when needed, and checking that the Windows OpenSSH service was installed and listening on port `22`.

### 10. Password SSH was confused by Windows account rules

When SSH asked for a password, it needed the real Windows account password, not the Windows Hello PIN. This caused confusion because Windows sign-in makes the PIN feel like the normal password. Once I understood that difference, it became much clearer why password-based SSH was failing.

### 11. The Linux private key did not exist at first

The website needed Linux to SSH into Windows, so the important private key had to live on the Linux server. At one point the config expected `~/.ssh/id_ed25519_web`, but that file was not there yet. The fix was to create the dedicated key pair on Linux and then place the matching public key on Windows.

### 12. Windows rejected the correct key because it was checking a different file

Even after Linux had the right key, Windows still rejected it. The cause was that the Windows user was in the `Administrators` group, so OpenSSH was checking the administrator key file instead of the normal user `authorized_keys` file. Once I moved the key to the correct place, Linux-to-Windows key authentication finally worked.

### 13. Manual SSH worked before the website worked

Another important lesson was that manual SSH success did not automatically mean the website would work. The site still needed the same host, user, key, path, and Python values in its config. So I had to copy the exact working values into the application settings.

### 14. The worker used the wrong PHP executable

The queue worker originally tried to use plain `php`, but the XAMPP setup needed `/opt/lampp/bin/php`. That meant the worker could fail before it even reached the queue logic. I fixed this by using the correct XAMPP CLI PHP path in the worker setup.

### 15. Failed jobs stayed failed after I fixed the code

Sometimes it looked like a fix had not worked, but the real issue was that the old jobs were still marked `failed`. The queue design does not retry those automatically. So part of the fix process was learning to re-queue or reset jobs after changing the code.

### 16. The queue failed with `''C:' is not recognized as an internal or external command`

This was one of the most useful error messages because it showed exactly where the system had reached. Linux had connected to Windows successfully, but the command being run on Windows was broken. The Unix-style quoting made Windows try to execute a damaged command beginning with `'C:'`. I fixed this by changing the remote command building so it used PowerShell with Windows-safe quoting rules.

### 17. Some apparent app failures were actually typing mistakes

One example was a request for `healthchecj.php` instead of `healthcheck.php`. The log showed the script was not found, but that did not mean the health check page itself was broken. This reminded me to separate genuine code errors from manual spelling mistakes in URLs and commands.

### 18. Tailscale showed `Funnel on` but `No serve config`

This meant the Funnel feature was enabled, but there was no actual local web service attached to it yet. So the site was still not being published. The fix was to set a proper backend target instead of assuming Funnel was complete just because it was switched on.

### 19. Tailscale also showed `listener already exists for port 443`

This error meant a previous Serve or Funnel setup was still using the public HTTPS listener. The answer was not to keep retrying the same command. The correct fix was to reset the old Serve and Funnel config first and then create the new mapping cleanly.

### 20. Tailscale required a localhost URL with an explicit port

When I tried using `http://127.0.0.1/`, Tailscale returned an error saying a port was required for a localhost target. That showed that the backend needed to be completely explicit. I corrected it by using `http://127.0.0.1:80`, which matched the XAMPP website.

### 21. The best final Funnel setup was to point it straight at `http://127.0.0.1:80`

The strongest final fix for the Tailscale problem was to clear old config and then attach Funnel directly to `http://127.0.0.1:80`. This made the local target clear and removed confusion. The expected result from that setup was a public `https://...ts.net` address that other people could use in a browser.

## Why I later simplified the frontend

### 1. The PHP version became more styled than I really needed

At first I used cards, panels, gradients, rounded corners, and other styling to make the prototype look organised. That was useful during early development, but later it felt too polished for the actual aim of the project.

### 2. I changed it back to simple HTML-style presentation

After the main logic was working, I stripped the website back to plain HTML presentation. This let the focus stay on the PHP logic, queue system, database behaviour, SSH connection, and deployment process instead of the visual design.

## Main lessons from the whole process

### 1. The real difficulty was systems integration

The hardest part of the website was not building the visible pages. The real challenge was connecting Linux XAMPP, PHP, MySQL, storage folders, CLI workers, SSH, Windows PowerShell, Python, and the forecasting scripts into one working system.

### 2. Server setup problems often looked like coding problems

Again and again, something that looked like a bug in the site turned out to be a missing permission, a broken config file, a wrong path, or a server setup issue. This taught me to check the environment as carefully as the code.

### 3. Clear error messages saved a lot of time

The most helpful moments were when I found a clear message such as `Undefined constant "DIR"`, `''C:' is not recognized...`, `No serve config`, or `listener already exists for port 443`. Each clear message narrowed the problem down and showed me which layer of the system I needed to fix next.

## Current state of the website work

The project now has a real PHP website rather than just a static mockup. It includes login, registration, ticker search, queue creation, saved graph viewing, health diagnostics, deployment-aware path handling, remote execution support, and documented public exposure steps. It has also been simplified visually so the main attention stays on functionality.

## Recommended next steps

The next sensible steps are to keep testing the full queue cycle on Linux, make sure the deployed `includes/jobs.php` file contains the Windows quoting fix, confirm the Tailscale Funnel setup with the correct localhost target, and continue improving logging and recovery for interrupted or stale jobs.

## Summary

The website started as a PHP/XAMPP interface for logging in, searching for tickers, queueing forecasts, and saving graphs. It then grew into a much larger deployment and debugging task across Linux and Windows. The most important results were not visual. They were the queue system, the SSH-driven training workflow, the database-backed history, the deployment fixes, and the detailed understanding of how all the parts of the system fit together. Rewriting the interface in a simpler HTML style supports that same goal by keeping the focus on how the system works.
