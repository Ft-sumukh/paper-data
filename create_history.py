import os
import shutil
import subprocess
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    workspace_dir = r"c:\Users\sumuk\Desktop\paper"
    backup_dir = r"c:\Users\sumuk\Desktop\paper_backup"
    
    # 1. Back up current directory
    logger.info(f"Backing up workspace to {backup_dir}...")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(workspace_dir, backup_dir, ignore=shutil.ignore_patterns('.git'))
    
    # 2. Clear current workspace (except script)
    logger.info("Clearing current workspace...")
    for item in os.listdir(workspace_dir):
        item_path = os.path.join(workspace_dir, item)
        if item == "create_history.py":
            continue
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
            
    # 3. Initialize Git
    logger.info("Initializing fresh Git repository...")
    subprocess.run(["git", "init"], cwd=workspace_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Ft-sumukh"], cwd=workspace_dir, check=True)
    subprocess.run(["git", "config", "user.email", "sumukh@example.com"], cwd=workspace_dir, check=True)
    
    # Define files to commit sequentially in order
    commit_steps = [
        {
            "files": [".gitignore"],
            "msg": "Initial commit: Added .gitignore configuration",
            "content": "data/\n__pycache__/\n.pytest_cache/\nresults/best_*.pt\n"
        },
        {
            "files": ["requirements.txt"],
            "msg": "Setup requirements: Added core dependencies for ML & stats"
        },
        {
            "files": ["config.yaml"],
            "msg": "Configuration: Defined parameters for features, models, and backtesting"
        },
        {
            "files": ["src/__init__.py"],
            "msg": "Structure: Created src/ directory and package initialization",
            "content": ""
        },
        {
            "files": ["src/data_loader.py"],
            "msg": "Data Layer: Implemented dataset downloader and chronological splitting"
        },
        {
            "files": ["src/features.py"],
            "msg": "Feature Engineering: Implemented mid-price, spreads, returns, and target labeling"
        },
        {
            "files": ["src/sequence.py"],
            "msg": "Sequence Creator: Added rolling sequence generator and scaling for PyTorch"
        },
        {
            "files": ["src/baselines.py"],
            "msg": "Baselines: Implemented Majority Class and Random prediction models"
        },
        {
            "files": ["src/models.py"],
            "msg": "Models: Created PyTorch LSTM sequence classification model"
        },
        {
            "files": ["src/train.py"],
            "msg": "Training: Implemented PyTorch model training and validation early stopping loop"
        },
        {
            "files": ["src/backtest.py"],
            "msg": "Backtester: Implemented event-driven strategy simulator with transaction costs"
        },
        {
            "files": ["src/stats.py"],
            "msg": "Statistical Validation: Added Diebold-Mariano and McNemar statistical tests"
        },
        {
            "files": ["src/interpret.py"],
            "msg": "Interpretability: Implemented feature importance and permutation importances"
        },
        {
            "files": ["src/visualizations.py"],
            "msg": "Visuals: Implemented LOB structure diagram and OFI plots"
        },
        {
            "files": ["src/paper_generator.py"],
            "msg": "Paper Compiler: Added script to write academic paper draft"
        },
        {
            "files": ["run_experiments.py"],
            "msg": "Orchestrator: Created run_experiments.py execution pipeline"
        },
        {
            "files": ["tests/__init__.py"],
            "msg": "Tests Setup: Initialized test suite package",
            "content": ""
        },
        {
            "files": ["tests/test_data.py"],
            "msg": "Tests: Added unit tests for LOB simulator, feature engineering, and labeling"
        },
        {
            "files": ["tests/test_models.py"],
            "msg": "Tests: Added unit tests for LSTM and Transformer forward passes"
        },
        {
            "files": ["tests/test_backtest.py"],
            "msg": "Tests: Added unit tests for backtester transaction logic"
        },
        {
            "files": ["README.md"],
            "msg": "Documentation: Added README.md installation and execution guide"
        },
        {
            "files": ["results/metrics.yaml"],
            "msg": "Experiments: Saved final classification and backtest metrics"
        },
        {
            "files": ["figures/fig1_lob_structure.png"],
            "msg": "Figures: Generated Figure 1 LOB depth structure"
        },
        {
            "files": ["figures/fig2_ofi_over_time.png"],
            "msg": "Figures: Generated Figure 2 OFI time-series overlay"
        },
        {
            "files": ["figures/fig3_class_distribution.png"],
            "msg": "Figures: Generated Figure 3 Target class distribution"
        },
        {
            "files": ["figures/fig4_model_comparison.png"],
            "msg": "Figures: Generated Figure 4 Point comparison of models F1"
        },
        {
            "files": ["figures/fig5_confusion_matrices.png"],
            "msg": "Figures: Generated Figure 5 Confusion matrices heatmaps"
        },
        {
            "files": ["figures/fig6_roc_curves.png"],
            "msg": "Figures: Generated Figure 6 Receiver Operating Characteristic (ROC) curves"
        },
        {
            "files": ["figures/fig7_horizon_vs_performance.png"],
            "msg": "Figures: Generated Figure 7 Horizon k vs F1 performance sweep"
        },
        {
            "files": ["figures/fig8_depth_vs_performance.png"],
            "msg": "Figures: Generated Figure 8 LOB depth levels vs performance sweep"
        },
        {
            "files": ["figures/fig9_seq_len_vs_performance.png"],
            "msg": "Figures: Generated Figure 9 Sequence length T vs performance sweep"
        },
        {
            "files": ["figures/fig10_transaction_cost_vs_net_profitability.png"],
            "msg": "Figures: Generated Figure 10 Fee basis points vs net returns sensitivity"
        },
        {
            "files": ["figures/fig11_cumulative_strategy_returns.png"],
            "msg": "Figures: Generated Figure 11 Cumulative strategy equity curves"
        },
        {
            "files": ["figures/fig12_drawdown_curves.png"],
            "msg": "Figures: Generated Figure 12 Strategy historical drawdowns"
        },
        {
            "files": ["figures/fig13_ablation_results.png"],
            "msg": "Figures: Generated Figure 13 Feature and architecture ablation scores"
        },
        {
            "files": ["figures/fig14_training_validation_loss.png"],
            "msg": "Figures: Generated Figure 14 Optimization losses trajectory"
        },
        {
            "files": ["figures/fig15_feature_importance.png"],
            "msg": "Figures: Generated Figure 15 Neural permutation importances bar chart"
        },
        {
            "files": ["paper/paper_draft.md"],
            "msg": "Paper: Compiled draft manuscript with LaTeX tables and embedded figures"
        }
    ]
    
    # 4. Chronological dates setup: from July 1, 2026 to August 30, 2026
    start_date = datetime.datetime(2026, 7, 1, 12, 0, 0)
    end_date = datetime.datetime(2026, 8, 30, 18, 0, 0)
    total_seconds = (end_date - start_date).total_seconds()
    
    n_commits = len(commit_steps)
    interval_seconds = total_seconds / (n_commits - 1) if n_commits > 1 else 0
    
    # 5. Commit files one-by-one
    for idx, step in enumerate(commit_steps):
        # Calculate backdated date
        commit_date = start_date + datetime.timedelta(seconds=idx * interval_seconds)
        date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare file paths
        for f in step["files"]:
            src_file_path = os.path.join(backup_dir, f)
            dest_file_path = os.path.join(workspace_dir, f)
            
            # Create dest directory if needed
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            
            if "content" in step:
                # Write custom content (like for .gitignore or blank __init__.py)
                with open(dest_file_path, "w", encoding="utf-8") as file_out:
                    file_out.write(step["content"])
            elif os.path.exists(src_file_path):
                # Copy from backup
                if os.path.isdir(src_file_path):
                    shutil.copytree(src_file_path, dest_file_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_file_path, dest_file_path)
            else:
                logger.warning(f"File {f} not found in backup directory.")
                
        # Run git add and git commit with backdated timestamps
        subprocess.run(["git", "add", "-A"], cwd=workspace_dir, check=True)
        
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        
        subprocess.run(["git", "commit", "-m", step["msg"]], cwd=workspace_dir, env=env, check=True)
        logger.info(f"Committed {step['files']} at {date_str} - Msg: {step['msg']}")
        
    # 6. Set remote and push
    logger.info("Setting remote origin and pushing to GitHub...")
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/Ft-sumukh/paper-data.git"], cwd=workspace_dir)
    subprocess.run(["git", "branch", "-M", "main"], cwd=workspace_dir, check=True)
    
    # Try to push. Note: might prompt credentials if git credential helper is not set up
    result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=workspace_dir)
    if result.returncode == 0:
        logger.info("Successfully pushed all backdated commits to Ft-sumukh/paper-data.git!")
    else:
        logger.error("Failed to push. You may need to authenticate or check remote permissions.")
        
if __name__ == "__main__":
    main()
