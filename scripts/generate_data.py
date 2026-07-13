import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_cohort(num_students=200, num_days=90, save_path="data/student_stress_dataset.csv"):
    print(f"Generating synthetic cohort: {num_students} students over {num_days} days...")
    
    np.random.seed(42)
    
    # Target dimensions
    DIMENSIONS = ["stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus"]
    
    # Set up directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_data = []
    
    # Semester exam schedule
    midterm_day = 45
    final_day = 88
    
    for i in range(num_students):
        student_id = f"std-{1000 + i}"
        
        # Student specific baselines (demographic/static metadata offsets)
        base_stress = np.random.uniform(0.2, 0.4)
        base_anxiety = np.random.uniform(0.15, 0.35)
        base_fatigue = np.random.uniform(0.2, 0.4)
        base_social = np.random.uniform(0.5, 0.8)
        base_academic = np.random.uniform(0.5, 0.8)
        base_burnout = np.random.uniform(0.05, 0.2)
        base_sleep = np.random.uniform(0.65, 0.85)
        base_mood = np.random.uniform(0.6, 0.8)
        base_resilience = np.random.uniform(0.55, 0.75)
        base_focus = np.random.uniform(0.55, 0.75)
        
        # Student coping parameter (resilience modifier)
        coping_factor = np.random.uniform(0.8, 1.2)
        
        # Generate raw timeline
        student_records = []
        for day in range(1, num_days + 1):
            # Calculate distance to upcoming exams
            if day <= midterm_day:
                days_to_midterms = float(midterm_day - day)
                days_to_finals = float(final_day - day)
            else:
                days_to_midterms = float(day - midterm_day) # past
                days_to_finals = float(final_day - day)
                
            # Academic pressure index: exponentially decaying distance to exam
            # High pressure near midterm (day 45) and final (day 88)
            dist_to_exam = min(abs(day - midterm_day), abs(day - final_day))
            academic_pressure = np.exp(-dist_to_exam / 7.0)
            
            # Weekend effect (weekly sinusoid d of week)
            day_of_week = day % 7
            is_weekend = 1.0 if day_of_week in [0, 6] else 0.0
            
            # 1. Stress: increases with academic pressure, decreases slightly on weekends
            stress = base_stress + 0.45 * academic_pressure - 0.1 * is_weekend + np.random.normal(0, 0.05)
            stress = max(0.0, min(1.0, stress * coping_factor))
            
            # 2. Anxiety: tracks stress closely
            anxiety = base_anxiety + 0.4 * academic_pressure + np.random.normal(0, 0.05)
            anxiety = max(0.0, min(1.0, anxiety * (1.1 - resilience_factor(base_resilience))))
            
            # 3. Fatigue: accumulates under academic pressure, falls on weekend (recovery)
            fatigue = base_fatigue + 0.4 * academic_pressure - 0.2 * is_weekend + np.random.normal(0, 0.05)
            fatigue = max(0.0, min(1.0, fatigue))
            
            # 4. Social: falls under academic pressure, rises on weekend
            social = base_social - 0.4 * academic_pressure + 0.25 * is_weekend + np.random.normal(0, 0.05)
            social = max(0.0, min(1.0, social))
            
            # 5. Academic performance: dips slightly under high stress/fatigue, rises on workload
            academic = base_academic + 0.1 * academic_pressure - 0.1 * stress + np.random.normal(0, 0.04)
            academic = max(0.0, min(1.0, academic))
            
            # 6. Burnout: slowly integrates stress over time
            # We use a running average of prior stress to simulate burnout buildup
            if len(student_records) > 0:
                prior_stress = np.mean([r["stress"] for r in student_records[-14:]])
                burnout = base_burnout + 0.5 * prior_stress * (day / num_days) + np.random.normal(0, 0.03)
            else:
                burnout = base_burnout
            burnout = max(0.0, min(1.0, burnout))
            
            # 7. Sleep: decreases with stress, increases on weekend
            sleep = base_sleep - 0.25 * stress + 0.15 * is_weekend + np.random.normal(0, 0.05)
            sleep = max(0.0, min(1.0, sleep))
            
            # 8. Mood: tracks sleep and social, drops with anxiety/stress
            mood = base_mood + 0.15 * sleep + 0.1 * social - 0.25 * stress + np.random.normal(0, 0.05)
            mood = max(0.0, min(1.0, mood))
            
            # 9. Resilience: stays relatively stable but dips slightly under high burnout
            resilience = base_resilience - 0.1 * burnout + np.random.normal(0, 0.03)
            resilience = max(0.0, min(1.0, resilience))
            
            # 10. Focus: dips with fatigue/anxiety, tracks academic dedication
            focus = base_focus - 0.2 * fatigue - 0.1 * anxiety + 0.1 * academic + np.random.normal(0, 0.05)
            focus = max(0.0, min(1.0, focus))
            
            rec = {
                "student_id": student_id,
                "day": day,
                "days_to_midterms": days_to_midterms,
                "days_to_finals": days_to_finals,
                "academic_pressure": academic_pressure,
                "is_weekend": is_weekend,
                "day_of_week": day_of_week,
                "stress": stress,
                "anxiety": anxiety,
                "fatigue": fatigue,
                "social": social,
                "academic": academic,
                "burnout": burnout,
                "sleep": sleep,
                "mood": mood,
                "resilience": resilience,
                "focus": focus
            }
            student_records.append(rec)
            
        # Data Augmentation: Jittering and time-warping
        # 1. Jittering (Adding minor Gaussian noise)
        # 2. Time-warping (shifting sequence steps slightly or scaling values)
        # We will apply this to a subset of records to expand the dataset
        all_data.extend(student_records)
        
        # Augment: Add a slightly altered clone (jittered trajectory) to double the data if needed
        # We have 200 * 90 = 18,000 records, which is already > 10k parameters! So no clone is strictly needed,
        # but let's add minor augmentation directly to the generated rows to simulate variety.
        
    df = pd.DataFrame(all_data)
    
    # Calculate rolling features
    print("Calculating rolling features...")
    df_list = []
    for sid, group in df.groupby("student_id"):
        group = group.sort_values("day")
        
        # Day of week sinusoid mappings
        group["sin_day"] = np.sin(2 * np.pi * group["day_of_week"] / 7.0)
        group["cos_day"] = np.cos(2 * np.pi * group["day_of_week"] / 7.0)
        
        # Rolling variance (moving std) of sleep count as specified in features list
        group["sleep_volatility_7d"] = group["sleep"].rolling(window=7, min_periods=1).std().fillna(0.0)
        group["stress_volatility_7d"] = group["stress"].rolling(window=7, min_periods=1).std().fillna(0.0)
        
        # Delta stress (7d)
        group["delta_stress_7d"] = group["stress"].diff(periods=7).fillna(0.0)
        
        # Sleep-stress ratio
        group["sleep_stress_ratio"] = group["sleep"] / (group["stress"] + 1e-5)
        
        df_list.append(group)
        
    df = pd.concat(df_list)
    df.to_csv(save_path, index=False)
    print(f"Dataset generated successfully! Total rows: {len(df)}. Saved to {save_path}")

def resilience_factor(res):
    # Mapping to help compute anxiety based on resilience
    return 0.3 * res

if __name__ == "__main__":
    generate_synthetic_cohort()
