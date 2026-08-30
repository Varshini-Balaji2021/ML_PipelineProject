from pipeline_builder import run_step, run_full_pipeline

class PipelineOrchestrator:
    def __init__(self):
        self.status = "Idle"

    def trigger_full_run(self):
        self.status = "Running"
        run_full_pipeline()
        self.status = "Completed"

    def trigger_single_stage(self, stage_name):
        self.status = f"Running {stage_name}"
        
        if stage_name == "Ingestion":
            run_step("Data Ingestion", "src/ingest.py")
            
        elif stage_name == "Preprocessing":
            run_step("Preprocessing", "src/preprocessing/preprocessing.py")
            
        elif stage_name == "Training":
            run_step("Model Training", "src/modeling/train.py")
            
        elif stage_name == "Evaluation":
            run_step("Model Evaluation", "src/evaluation/evaluate.py")
            
        self.status = "Idle"