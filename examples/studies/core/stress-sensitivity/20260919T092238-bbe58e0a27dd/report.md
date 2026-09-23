# Residual-price sensitivity

Proxy arms assume adverse residual liquidation relative to arrival; these are not observed prices or guaranteed worst-case bounds.

This is a post-hoc analysis of previously sealed stress evidence. The original statuses and files are preserved.

| Arm | Imputed rows | Available comparisons | Withheld comparisons |
| --- | ---: | ---: | ---: |
| raw | 0 | 16 | 8 |
| residual_stress_proxy_100bps | 93 | 24 | 0 |
| residual_stress_proxy_500bps | 93 | 24 | 0 |

Unavailable comparisons remain in each planned Holm family with p=1. Valid scenarios no longer inherit another scenario's INVALID veto.

The proxy applies 100/500 bps adverse arrival-relative prices to all unfilled quantity; it preserves actual filled shortfall and actual fees. It adds no invented terminal fee, and does not price unresolved orders.
