"""v1 statistical science: cluster-level inference, power, multiplicity and reference distributions."""
from .clustered import (ClusterTest, cluster_bootstrap, cluster_power, cluster_t, design_effect, feasible_clusters,
                        groups, holm_adjust, min_clusters_for, multiway_t, sign_flip)
from .distributions import betainc, nct_power, norm_cdf, norm_ppf, t_cdf, t_ppf, t_sf

__all__ = ["ClusterTest", "cluster_bootstrap", "cluster_power", "cluster_t", "design_effect", "feasible_clusters",
           "groups", "holm_adjust", "min_clusters_for", "multiway_t", "sign_flip", "betainc", "nct_power",
           "norm_cdf", "norm_ppf", "t_cdf", "t_ppf", "t_sf"]
