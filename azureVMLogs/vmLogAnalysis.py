#This was made with AI Assistance

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_all_data(pattern="VM*.jsonl"):
    """Load all JSONL files matching pattern"""
    dfs = []
    for file in sorted(Path(".").glob(pattern)):
        print(f"Loading {file}...")
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.json_normalize(data)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No data files found!")
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} records from {len(dfs)} files")
    return full_df

def prepare_data(df):
    """Clean and prepare data"""
    # Convert microseconds to milliseconds for readability
    df['wall_time_ms'] = df['wall_time_us'] / 1000
    df['cpu_time_user_ms'] = df['cpu_time_user_us'] / 1000
    df['cpu_time_system_ms'] = df['cpu_time_system_us'] / 1000
    df['cpu_time_total_ms'] = df['cpu_time_user_ms'] + df['cpu_time_system_ms']
    
    # Extract test config fields
    df['function'] = df['test_config.function'].astype(str)
    df['n'] = df['test_config.n'].astype(int)
    df['t'] = df['test_config.t'].astype(int)
    df['zkp_type'] = df['test_config.zkp_type'].fillna('None')
    df['run'] = df['test_config.run'].astype(int)
    df['t_ratio'] = df['t'] / df['n']
    
    # Filter successful runs
    df_success = df[df['status'] == 'SUCCESS'].copy()
    
    return df, df_success

def print_summary_stats(df, df_success):
    """Print overall summary statistics"""
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total records: {len(df)}")
    print(f"Successful: {len(df_success)} ({100*len(df_success)/len(df):.1f}%)")
    print(f"Failed: {len(df) - len(df_success)} ({100*(len(df)-len(df_success))/len(df):.1f}%)")
    print(f"\nVMs tested: {sorted(df['vm_profile'].unique())}")
    print(f"Architectures: {sorted(df['vm_arch'].unique())}")
    print(f"Functions tested: {sorted(df['function'].unique())}")
    print(f"Network sizes (n): {sorted(df['n'].unique())}")
    print(f"ZKP types: {sorted(df['zkp_type'].unique())}")

def aggregate_stats(df_success):
    """Calculate aggregate statistics by function, n, t, zkp_type"""
    group_cols = ['function', 'n', 't', 'zkp_type', 'vm_profile']
    
    agg_dict = {
        'wall_time_ms': ['mean', 'std', 'median', 'min', 'max'],
        'cpu_time_total_ms': ['mean', 'std', 'median'],
        'delta_rss_kb': ['mean', 'std', 'median', 'max'],
        'energy_estimate_joules': ['mean', 'std', 'median'],
        'output_size_bytes': ['mean', 'median'],
        'run': 'count'
    }
    
    stats = df_success.groupby(group_cols).agg(agg_dict).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    return stats

def plot_zkp_comparison(df_success, output_dir):
    """Compare ZKP types for ProofGen and ProofVerify"""
    zkp_funcs = ['ProofGen', 'ProofVerify']
    zkp_data = df_success[df_success['function'].isin(zkp_funcs) & (df_success['zkp_type'] != 'None')]
    
    if zkp_data.empty:
        print("No ZKP data found for comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ZKP Type Comparison (Bulletproof vs SNARK vs STARK)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('wall_time_ms', 'Wall Time (ms)', 0),
        ('cpu_time_total_ms', 'CPU Time (ms)', 1),
        ('delta_rss_kb', 'Memory Delta (KB)', 2)
    ]
    
    for func_idx, func in enumerate(zkp_funcs):
        func_data = zkp_data[zkp_data['function'] == func]
        
        for metric, label, col_idx in metrics:
            ax = axes[func_idx, col_idx]
            
            # Box plot
            sns.boxplot(data=func_data, x='zkp_type', y=metric, ax=ax, palette='Set2')
            ax.set_title(f'{func} - {label}', fontweight='bold')
            ax.set_xlabel('ZKP Type')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=0)
            
            # Add mean markers
            means = func_data.groupby('zkp_type')[metric].mean()
            ax.plot(range(len(means)), means, 'r*', markersize=15, label='Mean')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'zkp_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: zkp_comparison_boxplots.png")
    plt.close()
    
    # Statistical comparison table
    print("\n" + "="*80)
    print("ZKP TYPE PERFORMANCE COMPARISON")
    print("="*80)
    
    for func in zkp_funcs:
        func_data = zkp_data[zkp_data['function'] == func]
        print(f"\n{func}:")
        print("-" * 60)
        
        for zkp in ['Bulletproof', 'SNARK', 'STARK']:
            zkp_subset = func_data[func_data['zkp_type'] == zkp]
            if not zkp_subset.empty:
                print(f"\n  {zkp}:")
                print(f"    Wall Time:  {zkp_subset['wall_time_ms'].mean():.2f} ± {zkp_subset['wall_time_ms'].std():.2f} ms")
                print(f"    CPU Time:   {zkp_subset['cpu_time_total_ms'].mean():.2f} ± {zkp_subset['cpu_time_total_ms'].std():.2f} ms")
                print(f"    Memory:     {zkp_subset['delta_rss_kb'].mean():.2f} ± {zkp_subset['delta_rss_kb'].std():.2f} KB")
                print(f"    Energy:     {zkp_subset['energy_estimate_joules'].mean():.6f} ± {zkp_subset['energy_estimate_joules'].std():.6f} J")
                if 'output_size_bytes' in zkp_subset.columns:
                    print(f"    Proof Size: {zkp_subset['output_size_bytes'].mean():.0f} bytes")

def plot_network_scaling(df_success, output_dir):
    """Plot how functions scale with network size"""
    network_funcs = ['DKG1', 'DKG2', 'DKG3', 'PartialGen', 'AggCompute']
    scaling_data = df_success[df_success['function'].isin(network_funcs)]
    
    if scaling_data.empty:
        print("No network scaling data found")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Network Scaling Analysis (by Network Size)', fontsize=16, fontweight='bold')
    
    # Wall time scaling
    ax = axes[0, 0]
    for func in network_funcs:
        func_data = scaling_data[scaling_data['function'] == func]
        grouped = func_data.groupby('n')['wall_time_ms'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=func, linewidth=2)
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('Wall Time (ms)')
    ax.set_title('Wall Time vs Network Size', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # CPU time scaling
    ax = axes[0, 1]
    for func in network_funcs:
        func_data = scaling_data[scaling_data['function'] == func]
        grouped = func_data.groupby('n')['cpu_time_total_ms'].mean()
        ax.plot(grouped.index, grouped.values, marker='s', label=func, linewidth=2)
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('CPU Time (ms)')
    ax.set_title('CPU Time vs Network Size', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Memory scaling
    ax = axes[1, 0]
    for func in network_funcs:
        func_data = scaling_data[scaling_data['function'] == func]
        grouped = func_data.groupby('n')['delta_rss_kb'].mean()
        ax.plot(grouped.index, grouped.values, marker='^', label=func, linewidth=2)
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('Memory Delta (KB)')
    ax.set_title('Memory Usage vs Network Size', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Energy scaling
    ax = axes[1, 1]
    for func in network_funcs:
        func_data = scaling_data[scaling_data['function'] == func]
        grouped = func_data.groupby('n')['energy_estimate_joules'].mean()
        ax.plot(grouped.index, grouped.values, marker='d', label=func, linewidth=2)
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('Energy (Joules)')
    ax.set_title('Energy Consumption vs Network Size', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Heatmap: Function vs Network Size (Wall Time)
    ax = axes[2, 0]
    pivot = scaling_data.pivot_table(values='wall_time_ms', index='function', columns='n', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Wall Time (ms)'})
    ax.set_title('Wall Time Heatmap: Function × Network Size', fontweight='bold')
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('Function')
    
    # Scaling complexity analysis
    ax = axes[2, 1]
    for func in network_funcs:
        func_data = scaling_data[scaling_data['function'] == func]
        grouped = func_data.groupby('n')['wall_time_ms'].mean()
        
        # Fit polynomial to estimate complexity
        if len(grouped) > 3:
            log_n = np.log(grouped.index)
            log_time = np.log(grouped.values)
            coeffs = np.polyfit(log_n, log_time, 1)
            complexity = coeffs[0]
            
            fit_time = np.exp(coeffs[1]) * grouped.index ** complexity
            ax.plot(grouped.index, grouped.values, 'o-', label=f'{func} (O(n^{complexity:.2f}))', linewidth=2)
    
    ax.set_xlabel('Network Size (n)')
    ax.set_ylabel('Wall Time (ms)')
    ax.set_title('Computational Complexity Estimation', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_scaling.png', dpi=300, bbox_inches='tight')
    print(f"Saved: network_scaling.png")
    plt.close()

def plot_threshold_effects(df_success, output_dir):
    """Analyze threshold ratio effects"""
    threshold_data = df_success[df_success['t_ratio'].notna()]
    
    if threshold_data.empty:
        print("No threshold data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Threshold Ratio Effects', fontsize=16, fontweight='bold')
    
    # Wall time by threshold ratio
    ax = axes[0, 0]
    for func in threshold_data['function'].unique()[:5]:  # Top 5 functions
        func_data = threshold_data[threshold_data['function'] == func]
        grouped = func_data.groupby('t_ratio')['wall_time_ms'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=func, linewidth=2)
    ax.set_xlabel('Threshold Ratio (t/n)')
    ax.set_ylabel('Wall Time (ms)')
    ax.set_title('Wall Time vs Threshold Ratio', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory by threshold ratio
    ax = axes[0, 1]
    for func in threshold_data['function'].unique()[:5]:
        func_data = threshold_data[threshold_data['function'] == func]
        grouped = func_data.groupby('t_ratio')['delta_rss_kb'].mean()
        ax.plot(grouped.index, grouped.values, marker='s', label=func, linewidth=2)
    ax.set_xlabel('Threshold Ratio (t/n)')
    ax.set_ylabel('Memory Delta (KB)')
    ax.set_title('Memory vs Threshold Ratio', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heatmap: Function vs Threshold Ratio
    ax = axes[1, 0]
    pivot = threshold_data.pivot_table(values='wall_time_ms', index='function', columns='t_ratio', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Wall Time (ms)'})
    ax.set_title('Function × Threshold Ratio Heatmap', fontweight='bold')
    ax.set_xlabel('Threshold Ratio (t/n)')
    ax.set_ylabel('Function')
    
    # Distribution by threshold ratio
    ax = axes[1, 1]
    threshold_data_subset = threshold_data[threshold_data['function'].isin(['DKG3', 'PartialGen', 'AggCompute'])]
    sns.violinplot(data=threshold_data_subset, x='t_ratio', y='wall_time_ms', hue='function', ax=ax, split=False)
    ax.set_xlabel('Threshold Ratio (t/n)')
    ax.set_ylabel('Wall Time (ms)')
    ax.set_title('Performance Distribution by Threshold Ratio', fontweight='bold')
    ax.legend(title='Function')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_effects.png', dpi=300, bbox_inches='tight')
    print(f"Saved: threshold_effects.png")
    plt.close()

def plot_vm_comparison(df_success, output_dir):
    """Compare VM performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VM Performance Comparison', fontsize=16, fontweight='bold')
    
    # Overall performance by VM
    ax = axes[0, 0]
    vm_stats = df_success.groupby('vm_profile')['wall_time_ms'].mean().sort_values()
    vm_stats.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('VM Profile')
    ax.set_ylabel('Mean Wall Time (ms)')
    ax.set_title('Average Performance by VM', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # VM comparison box plot
    ax = axes[0, 1]
    top_vms = df_success['vm_profile'].value_counts().head(8).index
    vm_subset = df_success[df_success['vm_profile'].isin(top_vms)]
    sns.boxplot(data=vm_subset, x='vm_profile', y='wall_time_ms', ax=ax, palette='Set3')
    ax.set_xlabel('VM Profile')
    ax.set_ylabel('Wall Time (ms)')
    ax.set_title('Performance Distribution by VM', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log')
    
    # Memory usage by VM
    ax = axes[1, 0]
    vm_mem = df_success.groupby('vm_profile')['delta_rss_kb'].mean().sort_values()
    vm_mem.plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('VM Profile')
    ax.set_ylabel('Mean Memory Delta (KB)')
    ax.set_title('Memory Usage by VM', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Energy efficiency by VM
    ax = axes[1, 1]
    vm_energy = df_success.groupby('vm_profile')['energy_estimate_joules'].mean().sort_values()
    vm_energy.plot(kind='bar', ax=ax, color='green', alpha=0.7)
    ax.set_xlabel('VM Profile')
    ax.set_ylabel('Mean Energy (Joules)')
    ax.set_title('Energy Consumption by VM', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: vm_comparison.png")
    plt.close()

def plot_function_breakdown(df_success, output_dir):
    """Detailed breakdown by function"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Detailed Function Performance Analysis', fontsize=16, fontweight='bold')
    
    # Wall time by function
    ax = axes[0, 0]
    func_time = df_success.groupby('function')['wall_time_ms'].mean().sort_values()
    func_time.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_xlabel('Mean Wall Time (ms)')
    ax.set_ylabel('Function')
    ax.set_title('Average Wall Time by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # CPU time by function
    ax = axes[0, 1]
    func_cpu = df_success.groupby('function')['cpu_time_total_ms'].mean().sort_values()
    func_cpu.plot(kind='barh', ax=ax, color='lightcoral')
    ax.set_xlabel('Mean CPU Time (ms)')
    ax.set_ylabel('Function')
    ax.set_title('Average CPU Time by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Memory by function
    ax = axes[1, 0]
    func_mem = df_success.groupby('function')['delta_rss_kb'].mean().sort_values()
    func_mem.plot(kind='barh', ax=ax, color='lightgreen')
    ax.set_xlabel('Mean Memory Delta (KB)')
    ax.set_ylabel('Function')
    ax.set_title('Average Memory Usage by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Energy by function
    ax = axes[1, 1]
    func_energy = df_success.groupby('function')['energy_estimate_joules'].mean().sort_values()
    func_energy.plot(kind='barh', ax=ax, color='gold')
    ax.set_xlabel('Mean Energy (Joules)')
    ax.set_ylabel('Function')
    ax.set_title('Average Energy by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Variability analysis (CV)
    ax = axes[2, 0]
    func_cv = (df_success.groupby('function')['wall_time_ms'].std() / 
               df_success.groupby('function')['wall_time_ms'].mean()).sort_values()
    func_cv.plot(kind='barh', ax=ax, color='mediumpurple')
    ax.set_xlabel('Coefficient of Variation (CV)')
    ax.set_ylabel('Function')
    ax.set_title('Performance Variability by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Count of tests by function
    ax = axes[2, 1]
    func_count = df_success['function'].value_counts().sort_values()
    func_count.plot(kind='barh', ax=ax, color='orange')
    ax.set_xlabel('Number of Tests')
    ax.set_ylabel('Function')
    ax.set_title('Test Count by Function', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'function_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: function_breakdown.png")
    plt.close()

def plot_resource_efficiency(df_success, output_dir):
    """Analyze resource efficiency metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Resource Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # CPU efficiency (wall time vs cpu time)
    ax = axes[0, 0]
    sample = df_success.sample(min(1000, len(df_success)))
    ax.scatter(sample['wall_time_ms'], sample['cpu_time_total_ms'], alpha=0.5, s=20)
    ax.plot([0, sample['wall_time_ms'].max()], [0, sample['wall_time_ms'].max()], 'r--', label='Perfect efficiency')
    ax.set_xlabel('Wall Time (ms)')
    ax.set_ylabel('CPU Time (ms)')
    ax.set_title('CPU Efficiency: Wall Time vs CPU Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory efficiency (time vs memory)
    ax = axes[0, 1]
    sample = df_success[df_success['delta_rss_kb'] > 0].sample(min(1000, len(df_success)))
    ax.scatter(sample['wall_time_ms'], sample['delta_rss_kb'], alpha=0.5, s=20, c=sample['n'], cmap='viridis')
    ax.set_xlabel('Wall Time (ms)')
    ax.set_ylabel('Memory Delta (KB)')
    ax.set_title('Time vs Memory Trade-off', fontweight='bold')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Network Size (n)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Energy efficiency
    ax = axes[1, 0]
    func_efficiency = df_success.groupby('function').agg({
        'energy_estimate_joules': 'mean',
        'wall_time_ms': 'mean'
    })
    ax.scatter(func_efficiency['wall_time_ms'], func_efficiency['energy_estimate_joules'], s=100)
    for func in func_efficiency.index:
        ax.annotate(func, (func_efficiency.loc[func, 'wall_time_ms'], 
                           func_efficiency.loc[func, 'energy_estimate_joules']),
                   fontsize=8, alpha=0.7)
    ax.set_xlabel('Mean Wall Time (ms)')
    ax.set_ylabel('Mean Energy (Joules)')
    ax.set_title('Energy Efficiency by Function', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # I/O analysis
    ax = axes[1, 1]
    io_data = df_success[(df_success['disk_read_kb'] > 0) | (df_success['disk_write_kb'] > 0)]
    if not io_data.empty:
        func_io = io_data.groupby('function')[['disk_read_kb', 'disk_write_kb']].mean()
        func_io.plot(kind='bar', ax=ax, stacked=True)
        ax.set_xlabel('Function')
        ax.set_ylabel('Disk I/O (KB)')
        ax.set_title('Disk I/O by Function', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Read', 'Write'])
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No I/O data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Disk I/O Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resource_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: resource_efficiency.png")
    plt.close()

def plot_zkp_deep_dive(df_success, output_dir):
    """Deep dive into ZKP performance"""
    zkp_data = df_success[(df_success['zkp_type'] != 'None') & 
                          (df_success['function'].isin(['ProofGen', 'ProofVerify']))]
    
    if zkp_data.empty:
        print("No ZKP data for deep dive")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ZKP Deep Dive Analysis', fontsize=16, fontweight='bold')
    
    # Proof generation time distribution
    ax = axes[0, 0]
    proof_gen = zkp_data[zkp_data['function'] == 'ProofGen']
    for zkp in ['Bulletproof', 'SNARK', 'STARK']:
        subset = proof_gen[proof_gen['zkp_type'] == zkp]['wall_time_ms']
        if not subset.empty:
            ax.hist(subset, alpha=0.6, bins=20, label=zkp)
    ax.set_xlabel('Wall Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Proof Generation Time Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Proof verification time distribution
    ax = axes[0, 1]
    proof_ver = zkp_data[zkp_data['function'] == 'ProofVerify']
    for zkp in ['Bulletproof', 'SNARK', 'STARK']:
        subset = proof_ver[proof_ver['zkp_type'] == zkp]['wall_time_ms']
        if not subset.empty:
            ax.hist(subset, alpha=0.6, bins=20, label=zkp)
    ax.set_xlabel('Wall Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Proof Verification Time Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory comparison
    ax = axes[0, 2]
    zkp_mem = zkp_data.groupby(['zkp_type', 'function'])['delta_rss_kb'].mean().unstack()
    zkp_mem.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('ZKP Type')
    ax.set_ylabel('Mean Memory Delta (KB)')
    ax.set_title('Memory Usage: Generation vs Verification', fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Function')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Energy comparison
    ax = axes[1, 0]
    zkp_energy = zkp_data.groupby(['zkp_type', 'function'])['energy_estimate_joules'].mean().unstack()
    zkp_energy.plot(kind='bar', ax=ax, width=0.8, color=['#ff9999', '#66b3ff'])
    ax.set_xlabel('ZKP Type')
    ax.set_ylabel('Mean Energy (Joules)')
    ax.set_title('Energy: Generation vs Verification', fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Function')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Proof size comparison (if available)
    ax = axes[1, 1]
    proof_sizes = proof_gen[proof_gen['output_size_bytes'].notna()].groupby('zkp_type')['output_size_bytes'].mean()
    if not proof_sizes.empty:
        proof_sizes.plot(kind='bar', ax=ax, color='mediumpurple')
        ax.set_xlabel('ZKP Type')
        ax.set_ylabel('Proof Size (bytes)')
        ax.set_title('Average Proof Size', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No proof size data', ha='center', va='center', transform=ax.transAxes)
    
    # Speedup ratio (verification/generation)
    ax = axes[1, 2]
    speedup_data = []
    for zkp in ['Bulletproof', 'SNARK', 'STARK']:
        gen_time = proof_gen[proof_gen['zkp_type'] == zkp]['wall_time_ms'].mean()
        ver_time = proof_ver[proof_ver['zkp_type'] == zkp]['wall_time_ms'].mean()
        if gen_time > 0 and ver_time > 0:
            speedup_data.append({'ZKP': zkp, 'Speedup': gen_time / ver_time})
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        speedup_df.plot(x='ZKP', y='Speedup', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_xlabel('ZKP Type')
        ax.set_ylabel('Speedup Ratio (Gen/Ver)')
        ax.set_title('Verification Speedup vs Generation', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.axhline(y=1, color='r', linestyle='--', label='Equal time')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'zkp_deep_dive.png', dpi=300, bbox_inches='tight')
    print(f"Saved: zkp_deep_dive.png")
    plt.close()

def generate_summary_tables(df, df_success, output_dir):
    """Generate summary CSV tables"""
    
    # Overall summary by function
    func_summary = df_success.groupby('function').agg({
        'wall_time_ms': ['mean', 'std', 'median', 'min', 'max'],
        'cpu_time_total_ms': ['mean', 'std'],
        'delta_rss_kb': ['mean', 'std', 'max'],
        'energy_estimate_joules': ['mean', 'std'],
        'run': 'count'
    }).round(3)
    func_summary.columns = ['_'.join(col).strip('_') for col in func_summary.columns]
    func_summary.to_csv(output_dir / 'summary_by_function.csv')
    print(f"Saved: summary_by_function.csv")
    
    # ZKP comparison summary
    zkp_data = df_success[(df_success['zkp_type'] != 'None') & 
                          (df_success['function'].isin(['ProofGen', 'ProofVerify']))]
    if not zkp_data.empty:
        zkp_summary = zkp_data.groupby(['zkp_type', 'function']).agg({
            'wall_time_ms': ['mean', 'std', 'median'],
            'cpu_time_total_ms': ['mean', 'std'],
            'delta_rss_kb': ['mean', 'std'],
            'energy_estimate_joules': ['mean', 'std'],
            'output_size_bytes': 'mean',
            'run': 'count'
        }).round(3)
        zkp_summary.columns = ['_'.join(col).strip('_') for col in zkp_summary.columns]
        zkp_summary.to_csv(output_dir / 'zkp_comparison_summary.csv')
        print(f"Saved: zkp_comparison_summary.csv")
    
    # Network scaling summary
    network_funcs = ['DKG1', 'DKG2', 'DKG3', 'PartialGen', 'AggCompute']
    scaling_data = df_success[df_success['function'].isin(network_funcs)]
    if not scaling_data.empty:
        scaling_summary = scaling_data.groupby(['function', 'n']).agg({
            'wall_time_ms': ['mean', 'std'],
            'cpu_time_total_ms': ['mean', 'std'],
            'delta_rss_kb': ['mean', 'std'],
            'run': 'count'
        }).round(3)
        scaling_summary.columns = ['_'.join(col).strip('_') for col in scaling_summary.columns]
        scaling_summary.to_csv(output_dir / 'network_scaling_summary.csv')
        print(f"Saved: network_scaling_summary.csv")
    
    # VM comparison summary
    vm_summary = df_success.groupby(['vm_profile', 'vm_arch']).agg({
        'wall_time_ms': ['mean', 'std'],
        'cpu_time_total_ms': ['mean', 'std'],
        'delta_rss_kb': ['mean', 'std'],
        'energy_estimate_joules': ['mean', 'std'],
        'run': 'count'
    }).round(3)
    vm_summary.columns = ['_'.join(col).strip('_') for col in vm_summary.columns]
    vm_summary.to_csv(output_dir / 'vm_comparison_summary.csv')
    print(f"Saved: vm_comparison_summary.csv")
    
    # Failure analysis
    failures = df[df['status'] != 'SUCCESS']
    if not failures.empty:
        failure_summary = failures.groupby(['function', 'failure_reason']).size().reset_index(name='count')
        failure_summary.to_csv(output_dir / 'failure_analysis.csv', index=False)
        print(f"Saved: failure_analysis.csv")

def main():
    print("="*80)
    print("ZK-DISPHASIA BENCHMARK ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Load data
    df = load_all_data()
    df, df_success = prepare_data(df)
    
    # Print summary statistics
    print_summary_stats(df, df_success)
    
    # Generate all plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_zkp_comparison(df_success, output_dir)
    plot_zkp_deep_dive(df_success, output_dir)
    plot_network_scaling(df_success, output_dir)
    plot_threshold_effects(df_success, output_dir)
    plot_vm_comparison(df_success, output_dir)
    plot_function_breakdown(df_success, output_dir)
    plot_resource_efficiency(df_success, output_dir)
    
    # Generate summary tables
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80 + "\n")
    generate_summary_tables(df, df_success, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()