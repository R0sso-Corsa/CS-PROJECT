# Website Build Notes: Rationale, Thought Process, and Expected Difficulties

## Purpose of this document

This document explains the reasoning behind the website and server structure that was added for the project, together with the main difficulties that either appeared during implementation or are very likely to appear during deployment and testing. It is intended to act as a companion note for the web build rather than as user-facing documentation. The aim is to show why the site was structured the way it was, what trade-offs were made, which parts are already working conceptually, and which parts are most likely to cause trouble when the system is deployed on Linux/XAMPP and connected to the Windows training machine.

## Main design aim

The website needed to satisfy several practical requirements at the same time. It had to provide a simple set of pages for user interaction, allow a user to request a new prediction for a ticker, prevent multiple training jobs from running at the same time, keep old graphs available for later viewing, and connect a Linux/XAMPP PHP site to the training scripts on this PC through SSH. Because of that, the site could not just be a visual mockup. Even though the pages are intentionally basic, the structure had to reflect the real workflow of the final project.

## High-level thought process

### 1. Build the site around the real workflow, not just around page names

The first decision was that the pages should reflect the actual user journey through the system. That is why the site was divided into `index.php` for overview and entry, `login.php` for authentication, `search.php` for choosing a ticker, and `view.php` for either reviewing an older graph or checking job progress. This structure was chosen because it matches the problem description directly. Users need to log in, search for a ticker, decide whether to use an older graph or request a new one, and then have somewhere to monitor a result once it is queued or completed.

### 2. Avoid putting all logic directly inside page files

The next decision was to move most shared logic into `web/includes/`. This was done for maintainability. If the logic for database access, authentication, queue processing, layout rendering, and job management had been repeated inside each page file, the site would have become much harder to debug and extend. Separating that logic into reusable includes makes the structure easier to explain in coursework terms as decomposition and modular design, and it also makes future changes much less painful.

### 3. Use a queue table instead of trying to run training directly inside a page request

One of the most important design choices was not to let the search page run training immediately. Instead, the site writes a request into `prediction_jobs` with a status such as `queued`, `running`, `completed`, or `failed`. This was necessary because training is long-running and expensive. If the PHP page tried to run the full model directly during a browser request, the request might time out, Apache or PHP could kill the process, users would get poor feedback, and concurrency would be difficult to control. Using a queue makes the design much safer and much closer to how real systems work.

### 4. Only allow one active training job at a time

The project requirement said the system should check whether training is already ongoing and queue later requests until the current one is finished. To satisfy that, the queue worker checks whether a job is already marked `running`. If one is active, no additional queued job is claimed yet. This creates a simple but effective serial execution model. That choice was made instead of trying to support multiple jobs at once because the training pipeline is resource-intensive, the remote PC is the actual bottleneck, and a single-run model is easier both to reason about and to explain and test properly.

### 5. Store ticker history relationally instead of using literal sub-databases

The original request described “sub databases of each ticker,” but in practice MySQL or MariaDB is much better suited to a relational structure than to creating a separate database per ticker. The implemented structure therefore uses `tickers` as the parent record, `prediction_jobs` for queued or completed requests, `saved_graphs` for imported graph records, and `saved_graph_assets` for the actual PNG graph images. This achieves the same functional goal because each ticker has its own history, previous graphs can be retrieved easily, and jobs and graphs stay linked together. It also avoids the huge maintenance burden of creating separate databases or tables dynamically for every new ticker.

### 6. Add a remote helper script instead of making PHP guess file names

Another key decision was to add `web/tools/run_remote_prediction_job.py` on this PC. The reason was that once the Linux site SSHes into this machine and runs training, the server still needs to know which files were created, where the prediction CSV ended up, where the forecast CSV ended up, and where the plots were saved. If PHP had to guess those paths after the fact, the process would become fragile and difficult to maintain. The remote Python helper makes this much cleaner by running the training script, generating two headless PNG graphs, and returning a JSON manifest listing all important file paths. That makes the Linux server’s job much simpler.

## Why the pages were kept basic

The pages were intentionally built as basic but properly formatted pages instead of as a highly dynamic frontend. This was done because the project requirement prioritised flow and architecture over frontend complexity, PHP/XAMPP is already part of the expected deployment model, a lightweight interface is easier to get working quickly, and a simpler frontend keeps the attention on the queue, training, and database integration. The styling is still more polished than a raw classroom mockup, but it is not meant to be a final production interface yet.

## Difficulties that already came up

### 1. PHP could not be linted locally on this machine

The local environment here did not have `php` available on the PATH, which meant the PHP files could not be syntax-checked directly from this machine. The practical impact was that the PHP structure had to be reviewed carefully by inspection, while the Python helper could still be compiled and checked normally. Final PHP validation still needs to happen on the Linux/XAMPP machine. This is not a design flaw in the site itself, but it is a real implementation limitation that should be noted.

### 2. The existing web prototype was too rough to extend safely

The original `web` folder mainly contained a simple HTML login form, minimal CSS, and JavaScript with hardcoded credentials. That prototype was useful as evidence of early thinking, but it was not stable enough to become the real site. Because of that, the most practical route was to treat the new PHP structure as a clean scaffold instead of trying to preserve the old page logic. The result is that some early prototype ideas were effectively replaced and the new structure is much more realistic, but the downside is that the work became a partial rebuild rather than a tiny incremental edit.

### 3. File transfer between Linux and Windows is one of the hardest parts

The queue worker uses `ssh` and `scp`, which is the correct overall direction, but path handling between Linux and Windows is awkward. Difficult points include Windows paths like `C:\...` versus Unix paths like `/home/...`, the way OpenSSH on Windows exposes drive-letter paths, the need to quote paths safely when they contain spaces, and the need to make sure the Linux server can authenticate without manual password entry. This was partially accounted for in the code, but it is still one of the most likely places for deployment issues.

### 4. The training pipeline itself was not originally built as a web callback service

The existing Python forecasting scripts were written as local scripts, not as a networked API or web service. That means the website has to orchestrate them from the outside rather than talking to a native service endpoint. In practice, that means the queue worker needs to launch a script through SSH, the remote helper needs to wrap training and graph generation, and the error handling has to interpret command-line output rather than structured API responses. This is still workable, but it adds integration friction.

## Difficulties that will probably come up next

### 1. SSH authentication and permissions

This is probably the first real deployment issue that will appear. The Linux server may not have the correct private key, the Windows machine may not have OpenSSH server configured properly, the SSH user may not have permission to run Python or access the repository path, or `scp` may fail because of path or permission mismatches. This is the area most likely to need machine-specific debugging.

### 2. Background queue worker execution

The queue design assumes the Linux server can run `php` from the command line and can also spawn a background worker using `nohup`. Some XAMPP/Linux setups may not expose CLI PHP in the same environment as Apache, the Apache user may not have permission to spawn background processes, or the worker may start but fail silently if paths are wrong. If that happens, the queue records may remain stuck in `queued`.

### 3. Long-running training and timeout behavior

Even though training is not running inside the main page request anymore, the worker still has to wait for the remote process to finish. The remote training may take much longer than expected, the SSH session may drop, the worker may be interrupted by server restarts, and jobs may remain `running` if a process crashes partway through. That means a recovery strategy will probably be needed later, such as resetting stale jobs, storing more detailed logs, or adding retries.

### 4. Remote Python environment consistency

The helper script assumes that the remote machine has the correct Python interpreter, has all required packages installed, can run the existing forecasting pipeline without interactive prompts, and can render Matplotlib plots in headless mode. Even if the code itself is correct, the job can still fail because the wrong Python interpreter is used, dependencies are missing, GPU support differs between environments, or the script behaves differently under non-interactive execution.

### 5. Database storage size

The design stores graph PNGs in `saved_graph_assets` as BLOBs. This is useful because it keeps the graph record self-contained in the database, but it also has trade-offs. The database size will grow quickly if many graphs are saved, backups become larger, and queries may slow down if asset storage is not handled carefully. This is acceptable for a school project and a basic deployment, but if the system grows, storing files on disk and keeping only paths in the database might become more practical.

### 6. Authentication security

The current site includes a demo fallback account in the config for ease of testing. That is useful during development, but it is not suitable for a final secure deployment. Weak or hardcoded credentials, missing rate limiting, lack of a password reset flow, and session handling that is not hardened enough for real exposure are all possible issues. For the current stage that is acceptable as a prototype decision, but it would need improvement before being treated as a real public system.

### 7. User feedback during running jobs

At the moment, the site can display queued or running state, but it does not yet provide rich real-time progress updates. Users may not know how far along training is, a long-running job may feel stalled even if it is working, and there is no live polling dashboard yet. This is more of a usability limitation than an architectural problem, but it will likely come up once real testing starts.

## Why the current structure is still a strong starting point

Even with the expected difficulties, the current structure is a good base because it already covers the most important architectural ideas. It has a clear page flow, a separate login path, queue-based job control, ticker history, saved graph retrieval, a remote execution path, and a database-backed archive. That means the next stage is mostly about configuring the real deployment values, testing the SSH and file-transfer path, validating the PHP scripts on the Linux machine, and improving robustness, rather than redesigning the whole system from scratch.

## Recommended next implementation steps

The most practical next steps are to import the SQL schema into the Linux/XAMPP database, fill in the real values in `web/includes/config.php`, confirm SSH access manually from Linux to this PC before testing the queue, run `php -l` over the PHP files on the Linux machine, and then submit one ticker request and check the full path from queue to SSH to training to manifest to SCP import to database to view page. After that, the next sensible improvement would be adding recovery handling for stale `running` jobs and deciding later whether graphs should stay as database BLOBs or move to filesystem storage.

## Summary

The website was designed around the real forecasting workflow rather than around visual mock pages alone. The main thinking was to keep the frontend simple while making the backend structure realistic, with one queue, one running training job at a time, remote execution over SSH, database-backed ticker history, and a flow that supports both previous graph viewing and new graph creation. The main technical difficulty is not the HTML or PHP layout itself. It is the integration boundary between Linux/XAMPP, PHP worker logic, SSH and SCP, the Windows training machine, and the existing Python forecasting scripts. If those pieces are connected carefully, the rest of the system already has a solid foundation.
