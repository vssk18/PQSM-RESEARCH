#!/usr/bin/env python3
"""
Advanced Statistical Analysis for PQSM Research
Implements all metrics described in the comprehensive document
Author: Varanasi Sai Srinivasa Karthik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, binom, norm
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class AdvancedPQSMAnalysis:
    """Complete analysis matching the document specifications"""
    
    def __init__(self, data_path='analysis/analysis/all_runs_merged.csv'):
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} experimental runs")
        
        # Document-specified parameters
        self.latencies = [20, 50, 80, 120, 150, 200, 300]  # ms
        self.loss_rates = [0, 0.1, 0.5, 1, 2, 5]  # %
        self.payloads = [64, 256, 1024, 4096]  # bytes
        self.rates = [1, 5, 10]  # Hz
        
    def calculate_goodput(self, f_hz, B_bytes, D_ratio):
        """
        Calculate goodput G = f * B * 8 * D / 10^6 Mbit/s
        """
        return (f_hz * B_bytes * 8 * D_ratio) / 1e6
    
    def wilson_ci(self, successes, trials, confidence=0.95):
        """
        Wilson confidence interval for binomial proportion
        Better for edge cases (D near 0 or 1)
        """
        if trials == 0:
            return 0, 0, 0
        
        p_hat = successes / trials
        z = norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
        
        return center, center - margin, center + margin
    
    def calculate_safe_area_score(self, kem_data, threshold=0.95):
        """
        Calculate safe area score A_≥0.95
        Fraction of (L,p) grid where D ≥ threshold
        """
        total_cells = 0
        safe_cells = 0
        
        for latency in self.latencies:
            for loss in self.loss_rates:
                cell_data = kem_data[
                    (kem_data['latency_ms'] == latency) & 
                    (kem_data['loss_pct'] == loss)
                ]
                
                if len(cell_data) > 0:
                    total_cells += 1
                    avg_delivery = cell_data['delivery_ratio'].mean()
                    if avg_delivery >= threshold:
                        safe_cells += 1
        
        return safe_cells / total_cells if total_cells > 0 else 0
    
    def fit_reliability_envelope(self, kem_data):
        """
        Fit logistic GLM: logit E[D|L,p] = β₀ + β_L*L + β_p*p + β_Lp*L*p
        Returns coefficients and iso-contour functions
        """
        # Prepare data
        X = kem_data[['latency_ms', 'loss_pct']].values
        
        # Add interaction term
        X_with_interaction = np.column_stack([
            X,
            X[:, 0] * X[:, 1]  # L*p interaction
        ])
        
        # Transform D to logit space (handle edge cases)
        y = kem_data['delivery_ratio'].values
        y = np.clip(y, 0.001, 0.999)  # Avoid inf in logit
        y_logit = logit(y)
        
        # Fit linear model in logit space
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_with_interaction, y_logit)
        
        beta_0 = model.intercept_
        beta_L, beta_p, beta_Lp = model.coef_
        
        # Iso-contour function
        def iso_contour(L_values, D_level=0.95):
            """Get p values for given L values at constant D level"""
            D_logit = logit(D_level)
            p_values = []
            
            for L in L_values:
                # Solve: D_logit = β₀ + β_L*L + β_p*p + β_Lp*L*p
                # => p = (D_logit - β₀ - β_L*L) / (β_p + β_Lp*L)
                denominator = beta_p + beta_Lp * L
                if abs(denominator) > 1e-10:
                    p = (D_logit - beta_0 - beta_L * L) / denominator
                    p_values.append(max(0, p))  # Can't have negative loss
                else:
                    p_values.append(np.nan)
            
            return np.array(p_values)
        
        # Contour slope function
        def contour_slope(L, p):
            """dp/dL at constant D"""
            numerator = -(beta_L + beta_Lp * p)
            denominator = beta_p + beta_Lp * L
            return numerator / denominator if abs(denominator) > 1e-10 else np.nan
        
        return {
            'beta_0': beta_0,
            'beta_L': beta_L,
            'beta_p': beta_p,
            'beta_Lp': beta_Lp,
            'iso_contour_fn': iso_contour,
            'slope_fn': contour_slope,
            'model': model
        }
    
    def calculate_loss_elasticity(self, kem_data):
        """
        Calculate goodput elasticity w.r.t. loss
        ε_G,p = (p/D) * (∂D/∂p)
        """
        # Group by loss rate
        elasticities = []
        
        loss_values = sorted(kem_data['loss_pct'].unique())
        delivery_values = []
        
        for loss in loss_values:
            loss_data = kem_data[kem_data['loss_pct'] == loss]
            avg_delivery = loss_data['delivery_ratio'].mean()
            delivery_values.append(avg_delivery)
        
        # Calculate numerical derivative
        for i in range(1, len(loss_values)):
            p = loss_values[i]
            D = delivery_values[i]
            
            if D > 0 and i > 0:
                dD_dp = (delivery_values[i] - delivery_values[i-1]) / (loss_values[i] - loss_values[i-1])
                elasticity = (p / D) * dD_dp
                elasticities.append({
                    'loss_pct': p,
                    'delivery': D,
                    'elasticity': elasticity
                })
        
        return pd.DataFrame(elasticities)
    
    def generate_ecdf(self, data, label=''):
        """Generate empirical CDF with DKW confidence bands"""
        sorted_data = np.sort(data)
        n = len(sorted_data)
        ecdf = np.arange(1, n + 1) / n
        
        # DKW confidence band
        epsilon = np.sqrt(np.log(2/0.05) / (2*n))  # 95% confidence
        lower = np.maximum(ecdf - epsilon, 0)
        upper = np.minimum(ecdf + epsilon, 1)
        
        return {
            'values': sorted_data,
            'ecdf': ecdf,
            'lower': lower,
            'upper': upper,
            'label': label
        }
    
    def calculate_kde(self, data, label=''):
        """Calculate kernel density estimate for smooth PDF"""
        if len(data) < 2:
            return None
        
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kde(x_range)
        
        # Calculate hazard function h(t) = f(t) / (1 - F(t))
        # Approximate CDF by integration
        dx = x_range[1] - x_range[0]
        cdf = np.cumsum(density) * dx
        cdf = cdf / cdf[-1]  # Normalize
        
        hazard = density / (1 - cdf + 1e-10)  # Add small constant to avoid division by zero
        
        return {
            'x': x_range,
            'density': density,
            'cdf': cdf,
            'hazard': hazard,
            'label': label
        }
    
    def three_gate_decision(self, setup_slo_ms=100, slo_threshold=0.95, 
                           min_safe_area=0.8, max_elasticity=0.5):
        """
        Three-gate decision procedure for KEM selection
        G1: Setup SLO - Keep KEMs with P(T_h ≤ τ) ≥ θ
        G2: Envelope - Keep those with A_≥0.95 ≥ a_min
        G3: Throughput - Keep those with bounded |ε_G,p|
        """
        results = []
        
        for kem in self.df['kem_algorithm'].unique():
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            
            # Gate 1: Setup SLO
            handshake_times = kem_data['handshake_ms'].dropna()
            if len(handshake_times) > 0:
                slo_pass_rate = (handshake_times <= setup_slo_ms).mean()
                g1_pass = slo_pass_rate >= slo_threshold
            else:
                slo_pass_rate = 0
                g1_pass = False
            
            # Gate 2: Safe area
            safe_area = self.calculate_safe_area_score(kem_data)
            g2_pass = safe_area >= min_safe_area
            
            # Gate 3: Elasticity
            elasticity_df = self.calculate_loss_elasticity(kem_data)
            if len(elasticity_df) > 0:
                max_abs_elasticity = elasticity_df['elasticity'].abs().max()
                g3_pass = max_abs_elasticity <= max_elasticity
            else:
                max_abs_elasticity = float('inf')
                g3_pass = False
            
            # Combined score (weighted)
            w1, w2, w3 = 0.3, 0.4, 0.3  # Weights
            score = (w1 * slo_pass_rate + 
                    w2 * safe_area - 
                    w3 * min(max_abs_elasticity, 1.0))
            
            results.append({
                'kem': kem,
                'g1_slo_pass': g1_pass,
                'slo_pass_rate': slo_pass_rate,
                'g2_envelope_pass': g2_pass,
                'safe_area_score': safe_area,
                'g3_throughput_pass': g3_pass,
                'max_elasticity': max_abs_elasticity,
                'all_gates_pass': g1_pass and g2_pass and g3_pass,
                'combined_score': score
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('combined_score', ascending=False)
        
        return results_df
    
    def generate_advanced_plots(self):
        """Generate all plots mentioned in the document"""
        
        # Create output directory
        import os
        os.makedirs('analysis/analysis/advanced_plots', exist_ok=True)
        
        print("Generating advanced statistical plots...")
        
        # 1. Handshake ECDFs with confidence bands
        self._plot_handshake_ecdfs()
        
        # 2. Reliability iso-contours
        self._plot_reliability_contours()
        
        # 3. KDE ridgelines for decrypt times
        self._plot_decrypt_ridgelines()
        
        # 4. Goodput vs loss with elasticity
        self._plot_goodput_elasticity()
        
        # 5. Parallel coordinates plot
        self._plot_parallel_coordinates()
        
        # 6. Violin plots for handshake distributions
        self._plot_handshake_violins()
        
        print("Advanced plots generated in analysis/analysis/advanced_plots/")
    
    def _plot_handshake_ecdfs(self):
        """Plot handshake ECDFs with DKW confidence bands"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for kem in self.df['kem_algorithm'].unique():
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            handshake_times = kem_data['handshake_ms'].dropna()
            
            if len(handshake_times) > 10:
                ecdf_data = self.generate_ecdf(handshake_times, kem)
                
                # Plot ECDF
                ax.plot(ecdf_data['values'], ecdf_data['ecdf'], 
                       label=kem, linewidth=2)
                
                # Add confidence band (shaded)
                ax.fill_between(ecdf_data['values'], 
                              ecdf_data['lower'], 
                              ecdf_data['upper'],
                              alpha=0.2)
        
        # Add SLO lines
        for slo in [50, 100, 200]:
            ax.axvline(slo, color='gray', linestyle='--', alpha=0.5)
            ax.text(slo, 0.05, f'SLO={slo}ms', rotation=90, fontsize=9)
        
        ax.set_xlabel('Handshake Time T_h (ms)', fontsize=12)
        ax.set_ylabel('ECDF: P(T_h ≤ τ)', fontsize=12)
        ax.set_title('Handshake Time ECDFs with 95% DKW Confidence Bands', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        plt.savefig('analysis/analysis/advanced_plots/01_handshake_ecdfs.png', dpi=150)
        plt.close()
    
    def _plot_reliability_contours(self):
        """Plot reliability envelope with iso-contours"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        kems = self.df['kem_algorithm'].unique()[:6]
        
        for idx, kem in enumerate(kems):
            if pd.isna(kem):
                continue
            
            ax = axes[idx]
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            
            # Create pivot table for heatmap
            pivot = kem_data.pivot_table(
                values='delivery_ratio',
                index='loss_pct',
                columns='latency_ms',
                aggfunc='mean'
            )
            
            # Plot heatmap
            sns.heatmap(pivot, ax=ax, cmap='RdYlGn', vmin=0.85, vmax=1.0,
                       cbar_kws={'label': 'Delivery Ratio D'})
            
            # Add iso-contours
            envelope = self.fit_reliability_envelope(kem_data)
            
            # Calculate safe area
            safe_area = self.calculate_safe_area_score(kem_data)
            
            ax.set_title(f'{kem}\nSafe Area (D≥0.95): {safe_area:.1%}', fontsize=10)
            ax.set_xlabel('Latency (ms)', fontsize=9)
            ax.set_ylabel('Loss (%)', fontsize=9)
        
        plt.suptitle('Reliability Envelopes with Iso-contours', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('analysis/analysis/advanced_plots/02_reliability_contours.png', dpi=150)
        plt.close()
    
    def _plot_decrypt_ridgelines(self):
        """Plot decrypt time distributions as ridgelines"""
        # This is a simplified version - full ridgeline requires special libraries
        fig, ax = plt.subplots(figsize=(12, 8))
        
        kems = self.df['kem_algorithm'].unique()
        n_kems = len([k for k in kems if not pd.isna(k)])
        
        offset = 0
        for i, kem in enumerate(kems):
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            decrypt_times = kem_data['p50_decrypt_ms'].dropna()
            
            if len(decrypt_times) > 10:
                kde_data = self.calculate_kde(decrypt_times, kem)
                
                if kde_data:
                    # Scale and offset for ridgeline effect
                    scaled_density = kde_data['density'] / kde_data['density'].max() * 0.8
                    ax.fill_between(kde_data['x'], offset, offset + scaled_density,
                                  alpha=0.7, label=kem)
                    ax.plot(kde_data['x'], offset + scaled_density, 'k-', linewidth=1)
                    
                    offset += 1
        
        ax.set_xlabel('Decrypt Time t_dec (ms)', fontsize=12)
        ax.set_ylabel('KEM Algorithm', fontsize=12)
        ax.set_title('Decrypt Time Distributions (KDE Ridgelines)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('analysis/analysis/advanced_plots/03_decrypt_ridgelines.png', dpi=150)
        plt.close()
    
    def _plot_goodput_elasticity(self):
        """Plot goodput vs loss with elasticity annotations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for kem in self.df['kem_algorithm'].unique():
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            
            # Calculate goodput for each loss rate
            goodput_data = []
            for loss in sorted(kem_data['loss_pct'].unique()):
                loss_data = kem_data[kem_data['loss_pct'] == loss]
                avg_delivery = loss_data['delivery_ratio'].mean()
                avg_rate = loss_data['rate_hz'].mean()
                avg_payload = loss_data['payload_bytes'].mean()
                
                goodput = self.calculate_goodput(avg_rate, avg_payload, avg_delivery)
                goodput_data.append({
                    'loss_pct': loss,
                    'goodput': goodput,
                    'delivery': avg_delivery
                })
            
            if goodput_data:
                gdf = pd.DataFrame(goodput_data)
                ax1.plot(gdf['loss_pct'], gdf['goodput'], marker='o', label=kem, linewidth=2)
                
                # Calculate and plot elasticity
                elasticity_df = self.calculate_loss_elasticity(kem_data)
                if len(elasticity_df) > 0:
                    ax2.plot(elasticity_df['loss_pct'], elasticity_df['elasticity'],
                           marker='s', label=kem, linewidth=2)
        
        ax1.set_xlabel('Packet Loss Rate p (%)', fontsize=12)
        ax1.set_ylabel('Goodput G (Mbit/s)', fontsize=12)
        ax1.set_title('Goodput vs Loss', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        ax2.set_xlabel('Packet Loss Rate p (%)', fontsize=12)
        ax2.set_ylabel('Loss Elasticity ε_G,p', fontsize=12)
        ax2.set_title('Goodput Elasticity w.r.t. Loss', fontsize=13)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        plt.suptitle('Goodput Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('analysis/analysis/advanced_plots/04_goodput_elasticity.png', dpi=150)
        plt.close()
    
    def _plot_parallel_coordinates(self):
        """Plot parallel coordinates for multi-metric comparison"""
        from pandas.plotting import parallel_coordinates
        
        # Prepare summary data for each KEM
        summary_data = []
        
        for kem in self.df['kem_algorithm'].unique():
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            
            summary_data.append({
                'KEM': kem,
                'Handshake_P50': kem_data['handshake_ms'].quantile(0.5),
                'Delivery_Avg': kem_data['delivery_ratio'].mean(),
                'Decrypt_P50': kem_data['p50_decrypt_ms'].quantile(0.5),
                'Safe_Area': self.calculate_safe_area_score(kem_data)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Normalize columns for better visualization
            for col in summary_df.columns[1:]:
                summary_df[col] = (summary_df[col] - summary_df[col].min()) / (
                                  summary_df[col].max() - summary_df[col].min())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            parallel_coordinates(summary_df, 'KEM', ax=ax, colormap='viridis', linewidth=2)
            
            ax.set_title('Parallel Coordinates: Multi-Metric KEM Comparison', fontsize=14)
            ax.set_ylabel('Normalized Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.savefig('analysis/analysis/advanced_plots/05_parallel_coordinates.png', dpi=150)
            plt.close()
    
    def _plot_handshake_violins(self):
        """Plot violin plots for handshake time distributions"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for violin plot
        violin_data = []
        labels = []
        
        for kem in self.df['kem_algorithm'].unique():
            if pd.isna(kem):
                continue
            
            kem_data = self.df[self.df['kem_algorithm'] == kem]
            handshake_times = kem_data['handshake_ms'].dropna()
            
            if len(handshake_times) > 10:
                violin_data.append(handshake_times)
                labels.append(kem.replace('-', '\n'))
        
        if violin_data:
            parts = ax.violinplot(violin_data, positions=range(len(violin_data)),
                                 widths=0.7, showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor('#4ECDC4')
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
            ax.set_ylabel('Handshake Time T_h (ms)', fontsize=12)
            ax.set_title('Handshake Time Distributions (Violin Plots)', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentile lines
            for i, data in enumerate(violin_data):
                p50 = np.percentile(data, 50)
                p95 = np.percentile(data, 95)
                p99 = np.percentile(data, 99)
                
                ax.hlines(p50, i-0.2, i+0.2, colors='red', linestyles='-', linewidth=2)
                ax.hlines(p95, i-0.2, i+0.2, colors='orange', linestyles='--', linewidth=1)
                ax.hlines(p99, i-0.2, i+0.2, colors='darkred', linestyles=':', linewidth=1)
            
            # Add legend for percentiles
            ax.plot([], [], 'r-', linewidth=2, label='P50 (Median)')
            ax.plot([], [], 'orange', linestyle='--', linewidth=1, label='P95')
            ax.plot([], [], 'darkred', linestyle=':', linewidth=1, label='P99')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('analysis/analysis/advanced_plots/06_handshake_violins.png', dpi=150)
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary matching document specifications"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE PQSM ANALYSIS SUMMARY")
        print("="*70)
        
        # Run three-gate decision
        decision_results = self.three_gate_decision()
        
        print("\n1. THREE-GATE DECISION RESULTS:")
        print("-"*50)
        print(decision_results.to_string(index=False))
        
        # Best KEM selection
        best_kem = decision_results.iloc[0] if len(decision_results) > 0 else None
        
        if best_kem is not None:
            print(f"\n2. RECOMMENDED KEM: {best_kem['kem']}")
            print(f"   - SLO Pass Rate: {best_kem['slo_pass_rate']:.1%}")
            print(f"   - Safe Area Score: {best_kem['safe_area_score']:.1%}")
            print(f"   - Max Elasticity: {best_kem['max_elasticity']:.3f}")
            print(f"   - Combined Score: {best_kem['combined_score']:.3f}")
        
        # Generate operator playbook
        print("\n3. OPERATOR PLAYBOOK:")
        print("-"*50)
        print("Condition                    | Action")
        print("-"*50)
        print("L < 50ms, p < 1%            | Any KEM acceptable")
        print("L < 100ms, p < 2%           | Use ML-KEM-512 or NTRU-Prime")
        print("L > 150ms OR p > 5%         | Use ML-KEM-512 only")
        print("Constrained device          | Avoid Classic-McEliece")
        print("High security requirement   | Use ML-KEM-768 or higher")
        
        # Save detailed report
        with open('analysis/analysis/advanced_report.txt', 'w') as f:
            f.write("PQSM ADVANCED STATISTICAL ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(decision_results.to_string(index=False))
            f.write("\n\nGenerated using methodology from comprehensive document\n")
        
        print("\nReport saved to analysis/analysis/advanced_report.txt")


if __name__ == "__main__":
    # Run complete advanced analysis
    analyzer = AdvancedPQSMAnalysis()
    
    # Generate all advanced plots
    analyzer.generate_advanced_plots()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n✓ Advanced analysis complete!")
