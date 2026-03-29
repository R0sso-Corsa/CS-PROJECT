import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import sys
import os
from datetime import datetime


class AILearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Price Prediction - LSTM Model")
        self.root.geometry("950x750")
        self.root.configure(bg="#1e1e2e")

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(
            "Title.TLabel",
            background="#1e1e2e",
            foreground="#cdd6f4",
            font=("Segoe UI", 18, "bold"),
        )
        self.style.configure(
            "Header.TLabel",
            background="#1e1e2e",
            foreground="#cba6f7",
            font=("Segoe UI", 12, "bold"),
        )
        self.style.configure(
            "Label.TLabel",
            background="#1e1e2e",
            foreground="#bac2de",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "TEntry",
            fieldbackground="#313244",
            foreground="#cdd6f4",
            insertcolor="#cdd6f4",
        )
        self.style.configure("TCheckbutton", background="#1e1e2e", foreground="#cdd6f4")
        self.style.configure("TRadiobutton", background="#1e1e2e", foreground="#cdd6f4")
        self.style.configure(
            "TButton",
            background="#89b4fa",
            foreground="#1e1e2e",
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map("TButton", background=[("active", "#74c7ec")])
        self.style.configure(
            "Status.TLabel",
            background="#1e1e2e",
            foreground="#a6e3a1",
            font=("Segoe UI", 9),
        )

        self.process = None
        self.running = False

        self.create_widgets()

    def create_widgets(self):
        title = ttk.Label(self.root, text="AI Price Prediction", style="Title.TLabel")
        title.pack(pady=(20, 5))

        subtitle = ttk.Label(
            self.root,
            text="LSTM Neural Network for Financial Forecasting",
            style="Label.TLabel",
        )
        subtitle.pack(pady=(0, 20))

        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ticker Symbol Section
        ticker_frame = ttk.LabelFrame(left_frame, text="Ticker Symbol", padding=10)
        ticker_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(ticker_frame, text="Symbol:", style="Label.TLabel").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.ticker_var = tk.StringVar(value="BTC-GBP")
        self.ticker_entry = ttk.Entry(
            ticker_frame, textvariable=self.ticker_var, width=20, style="TEntry"
        )
        self.ticker_entry.grid(row=0, column=1, padx=(10, 0), pady=5)

        self.search_btn = ttk.Button(
            ticker_frame, text="Search", command=self.search_ticker, width=10
        )
        self.search_btn.grid(row=0, column=2, padx=(10, 0), pady=5)

        ttk.Label(
            ticker_frame,
            text="Examples: BTC-GBP, ETH-USD, AAPL, TSLA, GOOGL",
            style="Label.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # Device Settings
        device_frame = ttk.LabelFrame(left_frame, text="Device Settings", padding=10)
        device_frame.pack(fill=tk.X, pady=(0, 15))

        self.device_var = tk.StringVar(value="cpu")
        ttk.Radiobutton(
            device_frame, text="CPU", variable=self.device_var, value="cpu"
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            device_frame, text="GPU (CUDA)", variable=self.device_var, value="gpu"
        ).pack(anchor="w", pady=2)

        # Model Parameters
        params_frame = ttk.LabelFrame(left_frame, text="Model Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))

        param_configs = [
            ("Prediction Days:", "prediction_days", "30"),
            ("Future Days:", "future_days", "30"),
            ("Epochs:", "epochs", "150"),
            ("Batch Size:", "batch_size", "32"),
            ("Initial Dropout:", "init_dropout", "0.3"),
            ("Final Dropout:", "final_dropout", "0.1"),
            ("Monte Carlo Runs:", "mc_runs", "100"),
        ]

        self.param_vars = {}
        for i, (label, key, default) in enumerate(param_configs):
            ttk.Label(params_frame, text=label, style="Label.TLabel").grid(
                row=i, column=0, sticky="w", pady=4
            )
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            entry = ttk.Entry(params_frame, textvariable=var, width=12, style="TEntry")
            entry.grid(row=i, column=1, padx=(10, 0), pady=4)

        # Output Log
        output_frame = ttk.LabelFrame(right_frame, text="Output Log", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=65,
            height=28,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Consolas", 9),
            state="disabled",
            relief=tk.FLAT,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = ttk.Button(
            button_frame, text="Run Training", command=self.run_training
        )
        self.run_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(
            button_frame, text="Stop", command=self.stop_training, state="disabled"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_btn = ttk.Button(
            button_frame, text="Clear Log", command=self.clear_log
        )
        self.clear_btn.pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            left_frame,
            text="Ready - Enter parameters and click Run",
            style="Label.TLabel",
        )
        self.status_label.pack(pady=(15, 0))

    def log(self, message, color=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.configure(state="normal")
        if color:
            self.output_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.output_text.insert(tk.END, f"{message}\n", color)
        else:
            self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")

    def clear_log(self):
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state="disabled")

    def search_ticker(self):
        ticker = self.ticker_var.get().strip()
        if not ticker:
            messagebox.showwarning("Warning", "Please enter a ticker symbol to search")
            return
        self.log(f"Searching for: {ticker}", "info")
        try:
            import yfinance as yf

            result = yf.Search(ticker)
            if result and result.quotes:
                self.log(f"Found {len(result.quotes)} results:", "info")
                for i, q in enumerate(result.quotes[:10]):
                    name = q.get("shortName") or q.get("longName", "N/A")
                    self.log(
                        f"  {i + 1}. {q.get('symbol')} - {name} ({q.get('quoteType')})",
                        "result",
                    )
            else:
                self.log("No results found", "warning")
        except Exception as e:
            self.log(f"Search error: {e}", "error")

    def build_command(self):
        ticker = self.ticker_var.get().strip()
        if not ticker:
            return None

        script_path = os.path.join(
            os.path.dirname(__file__), "REWRITE", "pytorch_fixed.py"
        )

        cmd = [
            sys.executable,
            script_path,
            "--ticker",
            ticker,
            "--device",
            self.device_var.get(),
            "--prediction-days",
            self.param_vars["prediction_days"].get(),
            "--future-days",
            self.param_vars["future_days"].get(),
            "--epochs",
            self.param_vars["epochs"].get(),
            "--batch-size",
            self.param_vars["batch_size"].get(),
            "--init-dropout",
            self.param_vars["init_dropout"].get(),
            "--final-dropout",
            self.param_vars["final_dropout"].get(),
            "--mc-runs",
            self.param_vars["mc_runs"].get(),
        ]
        return cmd

    def run_training(self):
        ticker = self.ticker_var.get().strip()
        if not ticker:
            messagebox.showwarning("Warning", "Please enter a ticker symbol")
            return

        cmd = self.build_command()
        if not cmd:
            return

        self.running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(
            text="Training in progress...", foreground="#f9e2af"
        )
        self.log(f"Starting training for: {ticker}", "info")
        self.log(f"Command: {' '.join(cmd)}", "command")

        thread = threading.Thread(target=self.run_process, args=(cmd,), daemon=True)
        thread.start()

    def run_process(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in self.process.stdout:
                if not self.running:
                    break
                self.root.after(0, self.log, line.rstrip(), None)

            self.process.wait()
            if self.running:
                self.root.after(0, self.on_complete, self.process.returncode == 0)
            else:
                self.root.after(0, self.on_stopped)

        except Exception as e:
            self.root.after(0, self.on_error, str(e))

    def stop_training(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.log("Training stopped by user", "warning")

    def on_complete(self, success):
        self.running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if success:
            self.status_label.configure(text="Training complete!", foreground="#a6e3a1")
            self.log("Training completed successfully", "success")
        else:
            self.status_label.configure(text="Training failed", foreground="#f38ba8")
            self.log("Training completed with errors", "error")

    def on_stopped(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Training stopped", foreground="#f9e2af")

    def on_error(self, error):
        self.running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Error occurred", foreground="#f38ba8")
        self.log(f"Error: {error}", "error")


def main():
    root = tk.Tk()
    app = AILearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
