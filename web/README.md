# Web Forecast Portal

This folder now contains a basic PHP/XAMPP website scaffold for the project:

- `index.php`: home page and queue/archive overview
- `login.php`: simple login page with a demo fallback account
- `search.php`: search page where the user can view previous graphs or queue a new one
- `view.php`: viewing page for either a queued job or an imported graph
- `includes/`: shared config, database, auth, layout, and queue helpers
- `cli/process_queue.php`: background queue worker
- `tools/run_remote_prediction_job.py`: helper script that runs on this PC over SSH, launches training, and generates PNGs headlessly
- `sql/schema.sql`: MySQL schema for users, tickers, jobs, graphs, and graph assets

## Important setup steps

1. Import `web/sql/schema.sql` into your XAMPP MySQL or MariaDB instance.
2. Edit `web/includes/config.php`:
   - database credentials
   - SSH host/user/key
   - remote repo path on this PC
   - remote Python command if needed
3. Make sure the Linux server has:
   - `php` available on the CLI
   - `ssh` and `scp` installed
   - permission to write into `web/storage/`
4. Point Apache/XAMPP at the `web/` directory so `index.php` becomes the default entry page.

## Queue behavior

- New requests go into the `prediction_jobs` table with status `queued`.
- The background worker only claims a queued job if nothing is currently marked `running`.
- The worker sends one SSH command to this PC.
- The remote helper runs `REWRITE/separated/pytorch_train_cpp.py`, creates two headless graph PNGs, and returns a JSON manifest.
- The Linux server imports the PNG and CSV files back from this PC and stores the graph images as BLOBs in `saved_graph_assets`.

## About the ticker storage design

MySQL does not have true “sub-databases” for each ticker in the same way a filesystem has subfolders. The schema here uses a relational equivalent instead:

- `tickers` is the parent bucket
- `prediction_jobs` stores queued and completed requests per ticker
- `saved_graphs` stores graph records per ticker
- `saved_graph_assets` stores the actual image data for each saved graph

That gives you the same practical outcome while keeping the data much easier to query and maintain.
