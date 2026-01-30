import numpy as np
from scipy.stats import gaussian_kde

class Calibrator:
    """Converts raw scores to Likelihood Ratios: P(score|Unsafe) / P(score|Safe)"""
    def __init__(self):
        self.kde_safe = None
        self.kde_unsafe = None
        self.epsilon = 1e-12

    def fit(self, scores, labels):
        # labels: 0 = Safe, 1 = Unsafe
        safe_scores = scores[labels == 0]
        unsafe_scores = scores[labels == 1]
        
        # Add tiny noise to prevent singular matrix errors on constant data
        if np.std(safe_scores) == 0: safe_scores += np.random.normal(0, 1e-6, len(safe_scores))
        if np.std(unsafe_scores) == 0: unsafe_scores += np.random.normal(0, 1e-6, len(unsafe_scores))

        self.kde_safe = gaussian_kde(safe_scores)
        self.kde_unsafe = gaussian_kde(unsafe_scores)

    def get_lr(self, scores):
        scores = np.atleast_1d(scores)
        p_safe = self.kde_safe(scores)
        p_unsafe = self.kde_unsafe(scores)
        return p_unsafe / (p_safe + self.epsilon)

class ParetoCombiner:
    """
    Implements 'FullNPOneRegion': 
    1. Run Cheap Monitor.
    2. If score is in [t_start, t_end], pay cost to Run Expensive Monitor.
    3. Combine LRs and check against tau.
    """
    def __init__(self, cost_cheap=1.0, cost_exp=10.0, cost_intervention=50.0):
        self.c1 = cost_cheap
        self.c2 = cost_exp
        self.c_int = cost_intervention
        
        self.calib1 = Calibrator()
        self.calib2 = Calibrator()
        
        # Parameters to be learned via optimization
        self.t_start = -np.inf
        self.t_end = -np.inf
        self.tau = np.inf
        
        # --- CRITICAL ADDITION: History for Plotting ---
        self.frontier_history = [] 

    def fit_optimization(self, s1_val, s2_val, labels, budget_constraint):
        """
        Grid search to find optimal t_start, t_end, and tau that maximize Recall 
        subject to the budget constraint.
        """
        print("Calibrating monitors...")
        self.calib1.fit(s1_val, labels)
        self.calib2.fit(s2_val, labels)
        
        lr1 = self.calib1.get_lr(s1_val)
        lr2 = self.calib2.get_lr(s2_val)
        
        # Define grid for cheap monitor thresholds (quantiles)
        quantiles = np.linspace(0, 100, 50)
        thresholds = np.percentile(s1_val, quantiles)
        
        best_recall = 0
        best_params = (thresholds[0], thresholds[0], np.inf)
        
        # Reset history
        self.frontier_history = []

        print("Sweeping Pareto frontier...")
        for t_s in thresholds:
            for t_e in thresholds:
                if t_s >= t_e: continue
                
                # Identify Deferral Region (Uncertainty)
                defer_mask = (s1_val >= t_s) & (s1_val <= t_e)
                defer_rate = np.mean(defer_mask)
                
                # Check Sensing Cost (Average Compute Cost)
                avg_sensing_cost = self.c1 + (defer_rate * self.c2)
                remaining_for_int = budget_constraint - avg_sensing_cost
                
                if remaining_for_int <= 0: continue

                # Calculate Combined LRs
                final_lrs = np.where(defer_mask, lr1 * lr2, lr1)
                
                # Determine max interventions allowed by remaining budget
                max_int_rate = remaining_for_int / self.c_int
                
                if max_int_rate >= 1.0:
                    tau = 0
                else:
                    tau = np.quantile(final_lrs, 1 - max_int_rate)
                
                # Calculate Recall on Unsafe
                preds = final_lrs > tau
                recall = np.sum(preds & (labels == 1)) / np.sum(labels == 1)
                
                # --- SAVE DATA FOR THE GRAPH ---
                self.frontier_history.append({
                    "cost": avg_sensing_cost + (np.mean(preds) * self.c_int),
                    "recall": recall,
                    "defer_rate": defer_rate,
                    "thresholds": (t_s, t_e, tau)
                })
                # -------------------------------
                
                if recall > best_recall:
                    best_recall = recall
                    best_params = (t_s, t_e, tau)

        self.t_start, self.t_end, self.tau = best_params
        print(f"Optimization Complete. Recall: {best_recall:.3f} | Region: [{self.t_start:.3f}, {self.t_end:.3f}]")

    def predict(self, x, monitor1_func, monitor2_func):
        """
        Lazy evaluation inference.
        """
        s1 = monitor1_func(x)
        lr1 = self.calib1.get_lr([s1])[0]
        
        final_lr = lr1
        
        if self.t_start <= s1 <= self.t_end:
            s2 = monitor2_func(x)
            lr2 = self.calib2.get_lr([s2])[0]
            final_lr = lr1 * lr2
            
        return 1 if final_lr > self.tau else 0