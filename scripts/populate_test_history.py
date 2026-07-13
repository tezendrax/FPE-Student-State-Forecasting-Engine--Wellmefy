import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

def main():
    print("Populating sdt.db with 30 days of test history for student std-9874 and std-1001...")
    
    db_path = "../Digital Twin/sdt.db"
    if not os.path.exists(db_path):
        print(f"Error: sdt.db not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 1. Fetch the active encryption key
    c.execute("SELECT id, key_bytes FROM encryption_keys WHERE active = 1 LIMIT 1")
    row = c.fetchone()
    if not row:
        print("Error: No active encryption key found in sdt.db!")
        conn.close()
        return
        
    key_id, key_bytes = row
    fernet = Fernet(key_bytes)
    print(f"Found active key ID: {key_id}")
    
    # 2. Delete old history for test students
    c.execute("DELETE FROM twin_state_history WHERE student_id IN ('std-9874', 'std-1001', 'std-1002', 'std-1003')")
    c.execute("DELETE FROM digital_twin_states WHERE student_id IN ('std-9874', 'std-1001', 'std-1002', 'std-1003')")
    conn.commit()
    
    # Target dimensions
    DIMENSIONS = ["stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus"]
    
    students = {
        "std-9874": {
            "stress": 0.3, "anxiety": 0.25, "fatigue": 0.3, "social": 0.7,
            "academic": 0.6, "burnout": 0.15, "sleep": 0.75, "mood": 0.7,
            "resilience": 0.65, "focus": 0.7
        },
        "std-1001": {
            "stress": 0.5, "anxiety": 0.45, "fatigue": 0.5, "social": 0.5,
            "academic": 0.5, "burnout": 0.12, "sleep": 0.6, "mood": 0.55,
            "resilience": 0.5, "focus": 0.5
        },
        "std-1002": {
            "stress": 0.4, "anxiety": 0.35, "fatigue": 0.7, "social": 0.4,
            "academic": 0.55, "burnout": 0.18, "sleep": 0.4, "mood": 0.5,
            "resilience": 0.6, "focus": 0.45
        },
        "std-1003": {
            "stress": 0.2, "anxiety": 0.15, "fatigue": 0.2, "social": 0.8,
            "academic": 0.75, "burnout": 0.05, "sleep": 0.85, "mood": 0.82,
            "resilience": 0.8, "focus": 0.8
        }
    }
    
    now = datetime.utcnow()
    
    record_count = 0
    for student_id, baselines in students.items():
        print(f"Generating history for {student_id}...")
        
        # We generate 30 days of daily history ending today
        for i in range(30):
            day_offset = 30 - i
            timestamp = now - timedelta(days=day_offset)
            
            # Simulate daily progression with exam stress peaks around day 15 to 20
            # (which represents recent history for lookback)
            exam_pressure = np.exp(-abs(i - 18) / 4.0)
            
            stress = min(1.0, max(0.0, baselines["stress"] + 0.35 * exam_pressure + np.random.normal(0, 0.04)))
            anxiety = min(1.0, max(0.0, baselines["anxiety"] + 0.3 * exam_pressure + np.random.normal(0, 0.04)))
            fatigue = min(1.0, max(0.0, baselines["fatigue"] + 0.3 * exam_pressure + np.random.normal(0, 0.04)))
            sleep = min(1.0, max(0.0, baselines["sleep"] - 0.2 * exam_pressure + np.random.normal(0, 0.04)))
            mood = min(1.0, max(0.0, baselines["mood"] - 0.2 * exam_pressure + np.random.normal(0, 0.04)))
            social = min(1.0, max(0.0, baselines["social"] - 0.2 * exam_pressure + np.random.normal(0, 0.04)))
            academic = min(1.0, max(0.0, baselines["academic"] + 0.1 * exam_pressure + np.random.normal(0, 0.04)))
            # Profile-specific burnout trajectory calculations
            if student_id == "std-9874":
                burnout_val = baselines["burnout"] + 0.02 * i + np.random.normal(0, 0.02)
            elif student_id == "std-1001":
                burnout_val = baselines["burnout"] + 0.10 * exam_pressure + np.random.normal(0, 0.01)
            elif student_id == "std-1002":
                burnout_val = baselines["burnout"] + 0.005 * i + np.random.normal(0, 0.01)
            else: # std-1003 resilient
                burnout_val = baselines["burnout"] + 0.03 * exam_pressure + np.random.normal(0, 0.01)

            burnout = min(1.0, max(0.0, burnout_val))
            resilience = min(1.0, max(0.0, baselines["resilience"] - 0.05 * exam_pressure))
            focus = min(1.0, max(0.0, baselines["focus"] - 0.15 * exam_pressure))
            
            payload = {
                "stress": float(stress),
                "anxiety": float(anxiety),
                "fatigue": float(fatigue),
                "social": float(social),
                "academic": float(academic),
                "burnout": float(burnout),
                "sleep": float(sleep),
                "mood": float(mood),
                "resilience": float(resilience),
                "focus": float(focus)
            }
            
            payload_str = json.dumps(payload)
            encrypted_payload = fernet.encrypt(payload_str.encode("utf-8")).decode("utf-8")
            
            c.execute(
                """
                INSERT INTO twin_state_history (student_id, timestamp, encrypted_payload, key_id, trigger_source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, timestamp.strftime("%Y-%m-%d %H:%M:%S"), encrypted_payload, key_id, "api_lifestyle")
            )
            record_count += 1
            
        # Also populate current digital twin state
        c.execute(
            """
            INSERT INTO digital_twin_states (
                student_id, s_stress, s_anxiety, s_fatigue, s_social, s_academic,
                s_burnout, s_sleep, s_mood, s_resilience, s_focus,
                c_stress, c_anxiety, c_fatigue, c_social, c_academic,
                c_burnout, c_sleep, c_mood, c_resilience, c_focus,
                last_update_epoch, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                float(stress), float(anxiety), float(fatigue), float(social), float(academic),
                float(burnout), float(sleep), float(mood), float(resilience), float(focus),
                float(stress), float(anxiety), float(fatigue), float(social), float(academic),
                float(burnout), float(sleep), float(mood), float(resilience), float(focus),
                int(now.timestamp()), now.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        
    conn.commit()
    conn.close()
    print(f"Successfully inserted {record_count} historical records and active states in sdt.db!")

if __name__ == "__main__":
    main()
