# Dubai Hotel Revenue Management: results

Test period 2026-01-01 to 2026-06-30, 6 hotels, 318,560 room-nights.

## Occupancy forecast, 14 days before arrival

| Method | MAPE |
|---|---|
| Gradient boosting (calendar + events + pickup) | 3.1% |
| Additive pickup baseline | 5.6% |
| Same weekday last year | 10.9% |

## Pricing: legacy rate card vs dynamic pricing (same true demand)

| | Legacy | Dynamic |
|---|---|---|
| Revenue (AED) | 257,200,220 | 269,429,260 |
| Occupancy | 87.6% | 95.6% |
| ADR (AED) | 922 | 885 |
| RevPAR (AED) | 807 | 846 |

Revenue uplift: **4.8%** (AED 12,229,040 over six months), capturing 87% of the gain available with perfect knowledge of demand.

For comparison, a naive model (price as just another feature, sold-out days dropped) would have changed revenue by **-12.5%**: it under-estimates how price-sensitive guests are and over-prices.

## By hotel

| Hotel | Segment | RevPAR legacy | RevPAR dynamic | Uplift | Occ legacy | Occ dynamic |
|---|---|---|---|---|---|---|
| Palm Beach Resort | luxury_resort | 1259 | 1319 | 4.8% | 88% | 95% |
| Marina Luxury Hotel | luxury_city | 1009 | 1051 | 4.1% | 88% | 96% |
| Downtown Business Hotel | business | 701 | 734 | 4.7% | 86% | 95% |
| DIFC Executive Suites | business | 815 | 846 | 3.8% | 87% | 95% |
| Deira Midscale Inn | midscale | 327 | 349 | 6.7% | 88% | 96% |
| Al Barsha City Hotel | midscale | 362 | 388 | 7.4% | 87% | 97% |

## Price elasticity learned vs true

| Hotel | Learned | True |
|---|---|---|
| Palm Beach Resort | -1.05 | -1.1 |
| Marina Luxury Hotel | -1.21 | -1.25 |
| Downtown Business Hotel | -1.49 | -1.45 |
| DIFC Executive Suites | -1.24 | -1.3 |
| Deira Midscale Inn | -1.79 | -1.85 |
| Al Barsha City Hotel | -1.72 | -1.75 |