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

    def _generate_label(self, params):
        food = params.get('Number of food sources', '?')
        urge = params.get('Food attraction strength', '?')
        return f"Food={food}, Urge={urge}"

    def parse_file(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()

        params = self._extract_parameters(content)
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
        params = {}
        param_section = re.search(r'Experiment Parameters:(.*?)(?=\n=====)', content, re.DOTALL)
        if param_section:
            lines = param_section.group(1).strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()
        # Also look for 'Real time per frame' outside the parameter block
        rtpf_match = re.search(r'Real time per frame:\s*([0-9.]+)\s*ms', content)
        if rtpf_match:
            params['Real time per frame'] = rtpf_match.group(1) + ' ms'
        return params

    def _extract_csv_data(self, content, pattern, has_header=True):
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return pd.DataFrame()

        csv_text = match.group(1).strip()
        lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
        filtered_lines = [line for line in lines if not line.startswith('.') and ',' in line and not line.startswith('=')]

        if not filtered_lines:
            return pd.DataFrame()

        data_rows = []
        header = None
        for i, line in enumerate(filtered_lines):
            parts = [part.strip() for part in line.split(',')]
            if i == 0 and has_header:
                header = parts
            else:
                if header is None:
                    header = [f'col_{j}' for j in range(len(parts))]
                row = []
                for part in parts:
                    try:
                        row.append(float(part) if '.' in part else int(part))
                    except ValueError:
                        row.append(part)
                data_rows.append(row)

        return pd.DataFrame(data_rows, columns=header) if data_rows and header else pd.DataFrame()

    def _plot_time_series(self, key, xlabel, ylabel, title):
        plt.figure(figsize=(12, 8))
        plotted_any = False
        for filename, exp_data in self.data.items():
            df = exp_data['data'][key]
            if not df.empty and len(df.columns) >= 2:
                label = self._generate_label(exp_data['params'])
                ms_per_frame = exp_data['params'].get('Real time per frame', None)
                if ms_per_frame:
                    try:
                        seconds_per_frame = float(ms_per_frame.replace('ms', '').strip()) / 1000
                    except ValueError:
                        seconds_per_frame = 1.0
                else:
                    seconds_per_frame = 1.0

                x = df[df.columns[0]] * seconds_per_frame
                y = df[df.columns[1]]
                plt.plot(x, y, label=label, alpha=0.7, linewidth=2)
                plotted_any = True

        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        if plotted_any:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def plot_cohesion_before_food(self):
        return self._plot_time_series('cohesion_before', 'Frame', 'Cohesion', 'Cohesion History (Before Food Release)')

    def plot_cohesion_after_food(self):
        return self._plot_time_series('cohesion_after', 'Frame After Food Release', 'Cohesion', 'Cohesion History (After Food Release)')

    def plot_flock_count_before_food(self):
        return self._plot_time_series('flock_before', 'Frame', 'Flock Count', 'Flock Count History (Before Food Release)')

    def plot_flock_count_after_food(self):
        return self._plot_time_series('flock_after', 'Frame After Food Release', 'Flock Count', 'Flock Count History (After Food Release)')

    def _plot_distribution(self, key, xlabel, ylabel, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plotted_any = False
        data_list = []
        labels = []
        for filename, exp_data in self.data.items():
            df = exp_data['data'][key]
            if not df.empty and len(df.columns) >= 2:
                label = self._generate_label(exp_data['params'])
                data = df[df.columns[1]].values
                ax1.hist(data, alpha=0.6, bins=15, label=label, edgecolor='black')
                data_list.append(data)
                labels.append(label)
                plotted_any = True
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Frequency')
        ax1.set_title(title + ' (Histogram)')
        if plotted_any:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        if data_list:
            ax2.boxplot(data_list, labels=labels)
            ax2.set_ylabel(ylabel)
            ax2.set_title(title + ' (Box Plot)')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def plot_food_lifetime_distribution(self):
        return self._plot_distribution('food_lifetime', 'Food Lifetime (Frames)', 'Food Lifetime (Frames)', 'Food Lifetime Distribution')

    def plot_frames_to_consumption_distribution(self):
        return self._plot_distribution('frames_to_consumption', 'Frames to Consumption', 'Frames to Consumption', 'Frames to Consumption Distribution')

    def generate_all_plots(self, output_dir=None):
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
            for name, fig in plots.items():
                fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
                print(f"Saved {name}.png")
        return plots

    def print_summary(self):
        print(f"\n=== EXPERIMENT DATA SUMMARY ===")
        print(f"Number of files loaded: {len(self.data)}")
        for filename, exp_data in self.data.items():
            print(f"\nFile: {filename}")
            params = exp_data['params']
            for key in ['Number of particles (N)', 'Particle density (rho)', 'Simulation frames']:
                if key in params:
                    print(f"  {key.split('(')[0].strip()}: {params[key]}")
            print("  Data sections:", ', '.join([f"{k}({len(v)} rows)" if not v.empty else f"{k}(empty)" for k,v in exp_data['data'].items()]))

# Entry point
if __name__ == "__main__":
    analyzer = ExperimentDataAnalyzer()
    file_list = ['./output/experiment-food_amount_1-urge_1.txt',
                './output/experiment-food_amount_4-urge_1.txt',
                ]
    for file in file_list:
        analyzer.parse_file(file)
    analyzer.generate_all_plots(output_dir='plots')
    analyzer.print_summary()
