import pickle
import numpy as np
from pathlib import Path


def load_pickle(file_path):
    """Load pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compare_instances(file1_path, file2_path):
    """
    Compare two instance detailed results files and find scenarios where
    file1 performs better than file2 (file1 has lower metrics, file2 has higher metrics)
    
    Args:
        file1_path: Path to the first pickle file (lowrank results)
        file2_path: Path to the second pickle file (original results)
    
    Returns:
        Top 5 scenario IDs with largest performance gaps
    """
    # Load both files
    print(f"Loading file 1: {file1_path}")
    data1 = load_pickle(file1_path)
    
    print(f"Loading file 2: {file2_path}")
    data2 = load_pickle(file2_path)
    
    # Check if data is a list or dict and convert to dict if needed
    if isinstance(data1, list):
        print("File 1 is a list, converting to dict...")
        data1_dict = {}
        for item in data1:
            scenario_id = item.get('scenario_id', item.get('scenario', None))
            if scenario_id:
                data1_dict[scenario_id] = item
        data1 = data1_dict
    
    if isinstance(data2, list):
        print("File 2 is a list, converting to dict...")
        data2_dict = {}
        for item in data2:
            scenario_id = item.get('scenario_id', item.get('scenario', None))
            if scenario_id:
                data2_dict[scenario_id] = item
        data2 = data2_dict
    
    # Store comparison results
    comparisons = []
    
    # Get common scenario IDs
    scenarios1 = set(data1.keys())
    scenarios2 = set(data2.keys())
    common_scenarios = scenarios1.intersection(scenarios2)
    
    print(f"\nTotal scenarios in file 1: {len(scenarios1)}")
    print(f"Total scenarios in file 2: {len(scenarios2)}")
    print(f"Common scenarios: {len(common_scenarios)}")
    
    # Compare each common scenario
    for scenario_id in common_scenarios:
        result1 = data1[scenario_id]
        result2 = data2[scenario_id]
        
        # Extract minADE6 and minFDE6
        minADE6_1 = result1.get('minADE6', result1.get('minADE', None))
        minFDE6_1 = result1.get('minFDE6', result1.get('minFDE', None))
        
        minADE6_2 = result2.get('minADE6', result2.get('minADE', None))
        minFDE6_2 = result2.get('minFDE6', result2.get('minFDE', None))
        
        # Skip if any metric is missing
        if minADE6_1 is None or minFDE6_1 is None or minADE6_2 is None or minFDE6_2 is None:
            continue
        
        # Calculate differences (file2 - file1, we want positive values where file1 is better)
        ade_diff = minADE6_2 - minADE6_1
        fde_diff = minFDE6_2 - minFDE6_1
        
        # Only consider cases where file1 performs better (has lower metrics)
        if ade_diff > 0 and fde_diff > 0:
            # Calculate combined improvement (sum of differences)
            combined_diff = ade_diff + fde_diff
            
            comparisons.append({
                'scenario_id': scenario_id,
                'minADE6_file1': minADE6_1,
                'minADE6_file2': minADE6_2,
                'minFDE6_file1': minFDE6_1,
                'minFDE6_file2': minFDE6_2,
                'ade_diff': ade_diff,
                'fde_diff': fde_diff,
                'combined_diff': combined_diff
            })
    
    # Sort by ADE difference (descending)
    comparisons_ade = sorted(comparisons, key=lambda x: x['ade_diff'], reverse=True)
    top5_ade = comparisons_ade[:5]
    
    # Sort by FDE difference (descending)
    comparisons_fde = sorted(comparisons, key=lambda x: x['fde_diff'], reverse=True)
    top5_fde = comparisons_fde[:5]
    
    print(f"\nTotal scenarios where file1 performs better: {len(comparisons)}")
    
    # Print top 5 by minADE6
    print("\n" + "="*100)
    print("TOP 5 SCENARIOS WITH LARGEST minADE6 GAP (File1 better than File2)")
    print("="*100)
    
    for i, result in enumerate(top5_ade, 1):
        print(f"\nRank {i}:")
        print(f"  Scenario ID: {result['scenario_id']}")
        print(f"  minADE6 - File1: {result['minADE6_file1']:.4f}, File2: {result['minADE6_file2']:.4f}, Diff: {result['ade_diff']:.4f}")
        print(f"  minFDE6 - File1: {result['minFDE6_file1']:.4f}, File2: {result['minFDE6_file2']:.4f}, Diff: {result['fde_diff']:.4f}")
    
    print("\n" + "="*100)
    print("Top 5 Scenario IDs by minADE6:")
    for i, result in enumerate(top5_ade, 1):
        print(f"{i}. {result['scenario_id']}")
    print("="*100)
    
    # Print top 5 by minFDE6
    print("\n" + "="*100)
    print("TOP 5 SCENARIOS WITH LARGEST minFDE6 GAP (File1 better than File2)")
    print("="*100)
    
    for i, result in enumerate(top5_fde, 1):
        print(f"\nRank {i}:")
        print(f"  Scenario ID: {result['scenario_id']}")
        print(f"  minADE6 - File1: {result['minADE6_file1']:.4f}, File2: {result['minADE6_file2']:.4f}, Diff: {result['ade_diff']:.4f}")
        print(f"  minFDE6 - File1: {result['minFDE6_file1']:.4f}, File2: {result['minFDE6_file2']:.4f}, Diff: {result['fde_diff']:.4f}")
    
    print("\n" + "="*100)
    print("Top 5 Scenario IDs by minFDE6:")
    for i, result in enumerate(top5_fde, 1):
        print(f"{i}. {result['scenario_id']}")
    print("="*100)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY - All 10 Unique Scenario IDs:")
    all_scenario_ids = []
    for result in top5_ade:
        all_scenario_ids.append(result['scenario_id'])
    for result in top5_fde:
        if result['scenario_id'] not in all_scenario_ids:
            all_scenario_ids.append(result['scenario_id'])
    
    for i, scenario_id in enumerate(all_scenario_ids, 1):
        print(f"{i}. {scenario_id}")
    print("="*100)
    
    return {'top5_ade': top5_ade, 'top5_fde': top5_fde, 'all_unique_ids': all_scenario_ids}


def save_scenario_ids(results, output_dir):
    """Save scenario IDs to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "scenario_ids.txt"
    
    with open(output_file, 'w') as f:
        f.write("Top 5 Scenario IDs by minADE6:\n")
        for i, result in enumerate(results['top5_ade'], 1):
            f.write(f"{i}. {result['scenario_id']}\n")
        
        f.write("\nTop 5 Scenario IDs by minFDE6:\n")
        for i, result in enumerate(results['top5_fde'], 1):
            f.write(f"{i}. {result['scenario_id']}\n")
        
        f.write("\nSUMMARY - All Unique Scenario IDs:\n")
        for i, scenario_id in enumerate(results['all_unique_ids'], 1):
            f.write(f"{i}. {scenario_id}\n")
    
    print(f"\n\nScenario IDs saved to: {output_file}")


if __name__ == "__main__":
    # File paths
    file1 = "outputs/lowrank_r32_simplify_results_detail/seed44/20251021-170209/instance_detailed_results.pkl"
    file2 = "outputs/original_results_detail/seed44/20251021-165644/instance_detailed_results.pkl"
    
    # Convert to absolute paths
    file1_path = Path(file1)
    file2_path = Path(file2)
    
    # Run comparison
    results = compare_instances(file1_path, file2_path)
    
    # Save results
    output_dir = "outputs/viz_instance_compare"
    
    save_scenario_ids(results, output_dir)
