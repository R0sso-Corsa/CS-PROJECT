<?php

const APP_NAME = 'SignalStack Forecast Portal';
const APP_BASE_URL = '';

const DB_HOST = '127.0.0.1';
const DB_PORT = 3306;
const DB_NAME = 'cs_project';
const DB_USER = 'root';
const DB_PASS = '';

const DEMO_USERNAME = 'demo';
const DEMO_PASSWORD = 'demo123';

const REMOTE_SSH_HOST = 'YOUR_WINDOWS_PC_HOST';
const REMOTE_SSH_PORT = 22;
const REMOTE_SSH_USER = 'YOUR_WINDOWS_SSH_USER';
const REMOTE_SSH_KEY = '/home/your-linux-user/.ssh/id_ed25519';
const REMOTE_SSH_KNOWN_HOSTS = '';

const REMOTE_REPO_ROOT = 'C:/Users/paron/Desktop/Dev/CS_PROJECT';
const REMOTE_PYTHON = 'python';
const REMOTE_OUTPUT_ROOT = 'C:/Users/paron/Desktop/Dev/CS_PROJECT/REWRITE/separated/outputs/web_jobs';
const REMOTE_DEFAULT_DEVICE = 'gpu';
const REMOTE_DEFAULT_EPOCHS = 40;
const REMOTE_DEFAULT_BATCH_SIZE = 16;
const REMOTE_DEFAULT_PREDICTION_DAYS = 30;
const REMOTE_DEFAULT_FUTURE_DAYS = 30;
const REMOTE_DEFAULT_MC_RUNS = 100;
const PHP_CLI_BINARY = '/opt/lampp/bin/php';
const YFINANCE_SEARCH_PYTHON = 'python3';
const YFINANCE_SEARCH_ENV_PREFIX = 'env -u LD_LIBRARY_PATH';

define('WEB_ROOT', dirname(__DIR__));
define('WEB_STORAGE_ROOT', WEB_ROOT . '/storage');
define('GRAPH_IMPORT_ROOT', WEB_STORAGE_ROOT . '/graphs');
define('LOG_ROOT', WEB_STORAGE_ROOT . '/logs');
define('QUEUE_WORKER_SCRIPT', WEB_ROOT . '/cli/process_queue.php');
define('QUEUE_WORKER_LOG', LOG_ROOT . '/queue-worker.log');
define('QUEUE_WORKER_LOCK', LOG_ROOT . '/queue-worker.lock');
