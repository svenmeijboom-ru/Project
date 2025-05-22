import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

class ExperimentDataAnalyzer:
    def __init__(self):
        self.data = {}
        
    def parse_file(self, filepath):
        """Parse a single experiment data file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract experiment parameters
        params = self._extract_parameters(content)
        
        # Extract different data sections with more specific patterns
        data_sections = {
            'cohesion_before': self._extract_csv_data(content, r'Cohesion History \(before food release\):\s*\n(.*?)(?=\n\n|\nCohesion History \(after|\nFlock Count History|\Z)', has_header=True),
            'cohesion_after': self._extract_csv_data(content, r'Cohesion History \(after food release\):\s*\n(.*?)(?=\n\n|\nFlock Count History|\Z)', has_header=True),
            'flock_before': self._extract_csv_data(content, r'Flock Count History \(before food release\):\s*\n(.*?)(?=\n\n|\nFlock Count History \(after|\Z)', has_header=True),
            'flock_after': self._extract_csv_data(content, r'Flock Count History \(after food release\):\s*\n(.*?)(?=\n\n|\nFood Lifetime Data|\Z)', has_header=True),
            'food_lifetime': self._extract_csv_data(content, r'Food Lifetime Data:\s*\n(.*?)(?=\n\n|\nFrames from Creation|\Z)', has_header=True),
            'frames_to_consumption': self._extract_csv_data(content, r'Frames from Creation to Consumption:\s*\n(.*?)(?=\n\n|\nFrame Timestamps|\Z)', has_header=True)
        }
        
        filename = Path(filepath).stem
        self.data[filename] = {
            'params': params,
            'data': data_sections
        }
        
    def _extract_parameters(self, content):
        """Extract experiment parameters from the file"""
        params = {}
        param_section = re.search(r'Experiment Parameters:(.*?)(?=\n=====)', content, re.DOTALL)
        if param_section:
            lines = param_section.group(1).strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()
        return params
    
    def _extract_csv_data(self, content, pattern, has_header=True):
        """Extract CSV data using regex pattern"""
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return pd.DataFrame()
        
        csv_text = match.group(1).strip()
        lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
        
        # Filter out lines with dots (truncation indicators)
        filtered_lines = []
        for line in lines:
            if not line.startswith('.') and ',' in line and not line.startswith('='):
                filtered_lines.append(line)
        
        if not filtered_lines:
            return pd.DataFrame()
        
        # Parse CSV data
        data_rows = []
        header = None
        
        for i, line in enumerate(filtered_lines):
            parts = [part.strip() for part in line.split(',')]
            
            if i == 0 and has_header:
                # First line is header
                header = parts
            else:
                # Data row
                if header is None:
                    header = [f'col_{j}' for j in range(len(parts))]
                
                # Try to convert to numeric where possible
                row = []
                for part in parts:
                    try:
                        # Try int first, then float
                        if '.' in part:
                            row.append(float(part))
                        else:
                            row.append(int(part))
                    except ValueError:
                        row.append(part)
                data_rows.append(row)
        
        if data_rows and header:
            df = pd.DataFrame(data_rows, columns=header)
            print(f"Extracted {len(df)} rows with columns: {list(df.columns)}")
            return df
        return pd.DataFrame()
    
    def plot_cohesion_before_food(self):
        """Plot cohesion history before food release"""
        plt.figure(figsize=(12, 8))
        
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['cohesion_before']
            if not df.empty and len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                plt.plot(df[x_col], df[y_col], label=filename, alpha=0.7, linewidth=2)
                plotted_any = True
        
        plt.xlabel('Frame')
        plt.ylabel('Cohesion')
        plt.title('Cohesion History (Before Food Release)')
        if plotted_any:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_cohesion_after_food(self):
        """Plot cohesion history after food release"""
        plt.figure(figsize=(12, 8))
        
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['cohesion_after']
            if not df.empty and len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                plt.plot(df[x_col], df[y_col], label=filename, alpha=0.7, linewidth=2)
                plotted_any = True
        
        plt.xlabel('Frame After Food Release')
        plt.ylabel('Cohesion')
        plt.title('Cohesion History (After Food Release)')
        if plotted_any:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_flock_count_before_food(self):
        """Plot flock count history before food release"""
        plt.figure(figsize=(12, 8))
        
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['flock_before']
            if not df.empty and len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                plt.plot(df[x_col], df[y_col], label=filename, alpha=0.7, linewidth=2)
                plotted_any = True
        
        plt.xlabel('Frame')
        plt.ylabel('Flock Count')
        plt.title('Flock Count History (Before Food Release)')
        if plotted_any:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_flock_count_after_food(self):
        """Plot flock count history after food release"""
        plt.figure(figsize=(12, 8))
        
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['flock_after']
            if not df.empty and len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                plt.plot(df[x_col], df[y_col], label=filename, alpha=0.7, linewidth=2)
                plotted_any = True
        
        plt.xlabel('Frame After Food Release')
        plt.ylabel('Flock Count')
        plt.title('Flock Count History (After Food Release)')
        if plotted_any:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_food_lifetime_distribution(self):
        """Plot food lifetime data as histograms and box plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['food_lifetime']
            if not df.empty and len(df.columns) >= 2:
                lifetime_col = df.columns[1]
                ax1.hist(df[lifetime_col], alpha=0.6, bins=15, label=filename, edgecolor='black')
                plotted_any = True
        
        ax1.set_xlabel('Food Lifetime (Frames)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Food Lifetime Distribution')
        if plotted_any:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        lifetime_data = []
        labels = []
        for filename, exp_data in self.data.items():
            df = exp_data['data']['food_lifetime']
            if not df.empty and len(df.columns) >= 2:
                lifetime_col = df.columns[1]
                lifetime_data.append(df[lifetime_col].values)
                labels.append(filename)
        
        if lifetime_data:
            ax2.boxplot(lifetime_data, labels=labels)
            ax2.set_ylabel('Food Lifetime (Frames)')
            ax2.set_title('Food Lifetime Box Plot')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_frames_to_consumption_distribution(self):
        """Plot frames to consumption data as histograms and box plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data']['frames_to_consumption']
            if not df.empty and len(df.columns) >= 2:
                frames_col = df.columns[1]
                ax1.hist(df[frames_col], alpha=0.6, bins=15, label=filename, edgecolor='black')
                plotted_any = True
        
        ax1.set_xlabel('Frames to Consumption')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Frames to Consumption Distribution')
        if plotted_any:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        frames_data = []
        labels = []
        for filename, exp_data in self.data.items():
            df = exp_data['data']['frames_to_consumption']
            if not df.empty and len(df.columns) >= 2:
                frames_col = df.columns[1]
                frames_data.append(df[frames_col].values)
                labels.append(filename)
        
        if frames_data:
            ax2.boxplot(frames_data, labels=labels)
            ax2.set_ylabel('Frames to Consumption')
            ax2.set_title('Frames to Consumption Box Plot')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def generate_all_plots(self, output_dir=None):
        """Generate all plots and optionally save them"""
        plots = {
            'cohesion_before_food': self.plot_cohesion_before_food(),
            'cohesion_after_food': self.plot_cohesion_after_food(),
            'flock_count_before_food': self.plot_flock_count_before_food(),
            'flock_count_after_food': self.plot_flock_count_after_food(),
            'food_lifetime_distribution': self.plot_food_lifetime_distribution(),
            'frames_to_consumption_distribution': self.plot_frames_to_consumption_distribution()
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for plot_name, fig in plots.items():
                fig.savefig(os.path.join(output_dir, f'{plot_name}.png'), dpi=300, bbox_inches='tight')
                print(f"Saved {plot_name}.png")
        
        return plots
    
    def print_summary(self):
        """Print a summary of loaded data"""
        print(f"\n=== EXPERIMENT DATA SUMMARY ===")
        print(f"Number of files loaded: {len(self.data)}")
        
        for filename, exp_data in self.data.items():
            print(f"\nFile: {filename}")
            params = exp_data['params']
            if 'Number of particles (N)' in params:
                print(f"  Particles: {params['Number of particles (N)']}")
            if 'Particle density (rho)' in params:
                print(f"  Density: {params['Particle density (rho)']}")
            if 'Simulation frames' in params:
                print(f"  Frames: {params['Simulation frames']}")
            
            # Data availability with more details
            data_status = []
            for key, df in exp_data['data'].items():
                if not df.empty:
                    data_status.append(f"{key}({len(df)} rows)")
                else:
                    data_status.append(f"{key}(empty)")
            print(f"  Data sections: {', '.join(data_status)}")
            
            # Debug: Show first few rows of each non-empty dataset
            for key, df in exp_data['data'].items():
                if not df.empty:
                    print(f"    {key} sample:")
                    print(f"      Columns: {list(df.columns)}")
                    print(f"      First row: {df.iloc[0].tolist() if len(df) > 0 else 'None'}")
                else:
                    print(f"    {key}: No data extracted")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment data files and generate plots')
    parser.add_argument('files', nargs='+', help='Input .txt files to analyze')
    parser.add_argument('--output', '-o', help='Output directory for saving plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ExperimentDataAnalyzer()
    
    # Load files
    print("Loading files...")
    for filepath in args.files:
        if os.path.exists(filepath):
            print(f"  Loading: {filepath}")
            analyzer.parse_file(filepath)
        else:
            print(f"  Warning: File not found: {filepath}")
    
    # Print summary
    analyzer.print_summary()
    
    # Generate plots
    print("\nGenerating plots...")
    plots = analyzer.generate_all_plots(args.output)
    
    if args.show:
        plt.show()
    
    print("Analysis complete!")

# Example usage
if __name__ == "__main__":
    # If running as script
    #main()

    
    #Example programmatic usage:
    analyzer = ExperimentDataAnalyzer()
    analyzer.parse_file('./output/experiment-food_amount_4-urge_1.txt')
    analyzer.parse_file('./output/experiment-food_amount_9-urge_1.txt')
    analyzer.generate_all_plots(output_dir='plots')
    analyzer.print_summary()