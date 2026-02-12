import os
import shutil
import re
from ai_ml_plan_data import AI_ML_PLAN_DATA

def organize_pdfs():
    base_dir = "/Users/alejandropertinezgomez/Desktop/study_plan"
    
    # 1. Create Module Directories and Move Overview Files
    for i, module in enumerate(AI_ML_PLAN_DATA['modules']):
        month_num = i + 1
        # Create folder name like: Month_01_Fundamentals_of_AI_ML_Math
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', module['name'])
        folder_name = f"{clean_name}"
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
            
        # Move Module Overview PDF
        overview_filename = f"0{month_num}_Month_{month_num}_Overview.pdf"
        src = os.path.join(base_dir, overview_filename)
        dst = os.path.join(folder_path, overview_filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {overview_filename} to {folder_name}")

    # 2. Move Daily PDFs
    global_day_counter = 0
    for i, module in enumerate(AI_ML_PLAN_DATA['modules']):
        # Re-construct folder name to know where to move files
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', module['name'])
        folder_name = f"{clean_name}"
        folder_path = os.path.join(base_dir, folder_name)

        for week in module['weeks']:
            for day in week['days']:
                global_day_counter += 1
                
                # Reconstruct filename matching generate_ai_daily_pdfs.py logic
                clean_topic = re.sub(r'[^a-zA-Z0-9]', '_', day['topic'])
                clean_topic = re.sub(r'_+', '_', clean_topic).strip('_')
                filename_day = f"Day_{str(global_day_counter).zfill(3)}_{clean_topic}.pdf"
                
                src = os.path.join(base_dir, filename_day)
                dst = os.path.join(folder_path, filename_day)
                
                if os.path.exists(src):
                    shutil.move(src, dst)
                    # print(f"Moved {filename_day} to {folder_name}")
                else:
                    print(f"Warning: Could not find {src}")

if __name__ == "__main__":
    organize_pdfs()
