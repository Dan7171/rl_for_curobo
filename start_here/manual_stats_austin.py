import austin
from collections import defaultdict

def parse_austin_file(filename):
    """Parse Austin text format file"""
    stats = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split line into frames part and time
            if ' ' in line:
                frames_part, time_str = line.rsplit(' ', 1)
                try:
                    time = int(time_str)  # Time in microseconds
                except ValueError:
                    continue
                
                # Parse frames: P<pid>;T<tid>:<pid>;<frame1>;<frame2>;...
                if ';' in frames_part and ':' in frames_part:
                    # Split by ';' and skip the P<pid>;T<tid>:<pid> part
                    parts = frames_part.split(';')
                    frames = parts[1:]  # Skip P<pid> part
                    
                    # Skip T<tid>:<pid> part (first element after P<pid>)
                    if len(frames) > 0 and frames[0].startswith('T'):
                        frames = frames[1:]
                    
                    # Process each frame in the call stack
                    for frame in frames:
                        if ':' in frame:
                            # Clean up frame name
                            clean_frame = frame.replace('<frozen importlib._bootstrap>', 'importlib._bootstrap')
                            clean_frame = clean_frame.replace('<frozen importlib._bootstrap_external>', 'importlib._bootstrap_external')
                            
                            if clean_frame not in stats:
                                stats[clean_frame] = {'samples': 0, 'total_time': 0}
                            
                            stats[clean_frame]['samples'] += 1
                            stats[clean_frame]['total_time'] += time
    
    # Calculate averages (convert to seconds)
    for frame, data in stats.items():
        if data['samples'] > 0:
            data['avg_time_sec'] = (data['total_time'] / data['samples']) / 1_000_000  # Convert μs to seconds
            data['total_time_sec'] = data['total_time'] / 1_000_000  # Convert μs to seconds
        else:
            data['avg_time_sec'] = 0
            data['total_time_sec'] = 0
    
    return stats

# Parse the file
print("Parsing Austin file...")
stats = parse_austin_file('f.austin')

if stats:
    print(f"Found {len(stats)} unique functions")
    
    # Sort by average time per call (descending)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg_time_sec'], reverse=True)
    
    print("\nTop functions by AVERAGE time per call:")
    print("=" * 130)
    print(f"{'Function':<80} {'Avg (sec)':<12} {'Calls':<8} {'Total (sec)':<12} {'Total (ms)':<10}")
    print("=" * 130)
    
    for frame, data in sorted_stats[:30]:
        # Clean up frame name for display
        display_frame = frame.replace('/home/dan/rl_for_curobo/', '').replace('/home/dan/anaconda3/envs/env_isaacsim/lib/python3.10/site-packages/', '')
        if len(display_frame) > 80:
            display_frame = "..." + display_frame[-77:]
            
        total_ms = data['total_time_sec'] * 1000  # Convert to milliseconds for reference
        
        print(f"{display_frame:<80} {data['avg_time_sec']:<12.6f} {data['samples']:<8} {data['total_time_sec']:<12.3f} {total_ms:<10.1f}")
    
    print("\n" + "=" * 130)
    print("\nTop functions by TOTAL time:")
    print("=" * 130)
    print(f"{'Function':<80} {'Total (sec)':<12} {'Calls':<8} {'Avg (sec)':<12} {'Total (ms)':<10}")
    print("=" * 130)
    
    # Sort by total time
    sorted_by_total = sorted(stats.items(), key=lambda x: x[1]['total_time_sec'], reverse=True)
    
    for frame, data in sorted_by_total[:30]:
        display_frame = frame.replace('/home/dan/rl_for_curobo/', '').replace('/home/dan/anaconda3/envs/env_isaacsim/lib/python3.10/site-packages/', '')
        if len(display_frame) > 80:
            display_frame = "..." + display_frame[-77:]
            
        total_ms = data['total_time_sec'] * 1000
        
        print(f"{display_frame:<80} {data['total_time_sec']:<12.3f} {data['samples']:<8} {data['avg_time_sec']:<12.6f} {total_ms:<10.1f}")

else:
    print("No stats found")

# Load and analyze the austin file
try:
    stats = parse_austin_file('f.austin')
    
    if stats:
        # Sort by average time per call (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg_time_sec'], reverse=True)
        
        print("Functions sorted by average time per call:")
        print("-" * 100)
        print(f"{'Function':<60} {'Avg Time (sec)':<15} {'Calls':<10} {'Total (sec)':<15}")
        print("-" * 100)
        
        for func_name, data in sorted_stats[:30]:  # Top 30
            print(f"{func_name:<60} {data['avg_time_sec']:<15.6f} {data['samples']:<10} {data['total_time_sec']:<15.3f}")
    else:
        print("No stats found")
        
except Exception as e:
    print(f"Error parsing austin file: {e}")
    
    # Alternative: Try using austin2pstats if available
    print("\nTrying alternative method with austin2pstats...")
    try:
        import subprocess
        import pstats
        
        # Convert austin to pstats format
        result = subprocess.run(['austin2pstats', 'f.austin', 'temp.pstats'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Load pstats and show stats
            p = pstats.Stats('temp.pstats')
            p.sort_stats('cumulative')
            p.print_stats(30)
        else:
            print(f"austin2pstats failed: {result.stderr}")
            
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        print("\nPlease try: austin2speedscope f.austin")
        print("Or: austin-tui f.austin")