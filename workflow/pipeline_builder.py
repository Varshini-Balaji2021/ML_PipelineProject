from pathlib import Path
import subprocess

# PROJECT_ROOT is the parent of the 'workflow' folder (i.e., ML_Pipelineproject)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_step(step_name, script_rel_path):
    print(f"▶ Running {step_name}...")
    script_path = PROJECT_ROOT / script_rel_path
    
    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        return False
        
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if res.returncode == 0:
        print(f"✅ {step_name} completed successfully!")
        print(res.stdout)
        return True
    else:
        print(f"❌ {step_name} failed.")
        print(res.stderr)
        return False

def run_full_pipeline():
    """Executes the end-to-end machine learning pipeline step-by-step."""
    print("🚀 Starting Full ML Pipeline Execution...")
    
    # 1. Ingestion
    if not run_step("Ingestion", "src/ingest.py"):
        return
        
    # 2. Preprocessing & Splitting
    if not run_step("Preprocessing", "src/preprocessing/preprocessing.py"):
        return
        
    # 3. Model Training
    if not run_step("Model Training", "src/modeling/train.py"):
        return
        
    # 4. Evaluation
    if not run_step("Evaluation", "src/evaluation/evaluate.py"):
        return
        
    print("🎉 Full Pipeline Executed Successfully!")

if __name__ == "__main__":
    run_full_pipeline()