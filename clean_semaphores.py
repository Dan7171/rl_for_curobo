#!/usr/bin/env python3
import subprocess
import os

def clean_semaphores():
    """Clean up leaked semaphores from previous Isaac Sim runs"""
    username = os.getenv('USER')
    
    try:
        # Get list of semaphores owned by current user
        result = subprocess.run(['ipcs', '-s'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        semaphore_ids = []
        for line in lines:
            if username in line:
                parts = line.split()
                if len(parts) >= 2:
                    sem_id = parts[1]
                    semaphore_ids.append(sem_id)
        
        if semaphore_ids:
            print(f"Found {len(semaphore_ids)} semaphores to clean: {semaphore_ids}")
            
            # Remove each semaphore
            for sem_id in semaphore_ids:
                try:
                    subprocess.run(['ipcrm', '-s', sem_id], check=True)
                    print(f"✅ Removed semaphore {sem_id}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to remove semaphore {sem_id}: {e}")
            
            print("✅ Semaphore cleanup completed")
        else:
            print("✅ No semaphores found to clean")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    clean_semaphores()
