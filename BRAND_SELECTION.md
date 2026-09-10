# Brand Selection Analysis

Method: `scripts/01_dataset_overview.py`, `scripts/02_brand_selection.py`, `scripts/03_deflection_check.py`.
All numbers below are computed directly from `data/raw/twcs.csv` (2,811,774 rows after
pandas parsing of embedded newlines/quotes; raw `wc -l` reports 3,002,524 because some
tweet text fields contain literal newlines inside quoted CSV fields).

## Candidates considered
Top 14 brand accounts by outbound message volume: AmazonHelp, AppleSupport, Uber_Support,
SpotifyCares, Delta, AmericanAir, TMobileHelp, comcastcares, British_Airways, SouthwestAir,
XboxSupport, hulu_support, AskPlayStation, VerizonSupport.

## Selection criteria (measured, not assumed)
1. **Volume** — enough customer↔brand reply pairs to support retrieval + a 150-250 example
   golden set with room for stratification.
2. **Deflection rate** — % of brand replies that are pure "please DM us" boilerplate with no
   actual resolution content. High deflection means there is *nothing to ground responses in*
   publicly — the real resolution happened in a DM we don't have. This is the single most
   important criterion for this assignment because grounding is the core differentiator.
3. **Conversation depth / freshness** — average thread depth and % of replies that are to a
   fresh (root) complaint vs. a deep sub-thread, as a proxy for conversational complexity.
4. **Noise** — duplicate/templated-reply rate in customer messages, URL rate.

## Results

| Brand | Cust↔Brand pairs | Fresh-root % | Avg thread depth | Dup rate % | **DM-deflection %** | Avg reply length (words) |
|---|---|---|---|---|---|---|
| **AmazonHelp** | **168,814** | 50.1 | 2.92 | 9.4 | **0.6** | 19.8 |
| Delta | 42,114 | 67.6 | 1.87 | 15.5 | 16.5 | 17.9 |
| AmericanAir | 36,531 | 67.1 | 1.85 | 0.6 | 16.8 | 18.4 |
| SpotifyCares | 43,092 | 62.5 | 1.84 | 4.7 | 30.8 | 22.1 |
| Uber_Support | 56,160 | 71.3 | 1.73 | 3.0 | 35.9 | 19.2 |
| AppleSupport | 106,646 | 70.0 | 1.70 | 1.7 | 52.5 | 22.7 |
| comcastcares | 32,921 | 72.4 | 1.67 | 9.1 | 71.5 | 24.0 |

(Full 14-candidate table: `artifacts/data_quality/brand_candidates.csv`,
deflection rates: `artifacts/data_quality/deflection_rates.csv`)

## Selected brand: **AmazonHelp**

**Why the evidence supports it:**
- Highest volume by a wide margin (168.8k customer→brand reply pairs) — comfortably supports
  a stratified 150-250 example golden set *and* leaves the rest for a leakage-free retrieval
  index and baseline training data.
- **By far the lowest DM-deflection rate (0.6% vs 16.5-71.5% for the rest).** This is the
  deciding factor: for AmazonHelp, the historical reply text in the dataset actually *contains*
  the resolution (order status explanation, return/refund policy statement, troubleshooting
  step, escalation to a form/link), rather than being a dead-end "please DM us." That is a
  hard requirement for a system whose core claim is "grounded in how the brand historically
  resolved similar issues" — you cannot ground on a deflection.
- Amazon's support surface spans several genuinely distinct issue families (order/shipping
  status, returns & refunds, billing/payments, device support for Kindle/Echo/Fire TV,
  digital content/Prime Video, account/login, Prime membership) — enough real intent
  diversity to justify an 8-12 class taxonomy without forcing artificial splits.
- Deeper average thread structure (2.92 vs ~1.7-1.9 for most others) gives more multi-turn
  context to work with for the "conversation/context preparation" pipeline stage.

**Risks / biases this introduces (documented honestly):**
- Amazon is an extremely heterogeneous business (retail marketplace + devices + digital
  services + payments). A single 8-12 intent taxonomy will necessarily be a simplification;
  we scope it to the highest-frequency issue families and use an explicit "other/unclear"
  bucket rather than pretending full coverage (see `INTENT_TAXONOMY.md`).
- AmazonHelp is the single most-used brand in prior public work built on this dataset, which
  is a double-edged sword: it makes our numbers easier to sanity-check against community
  priors, but means our "novel" empirical contribution is smaller than picking an unstudied
  brand.
- The 9.4% duplicate rate in customer messages (vs. e.g. 0.6% for AmericanAir) suggests more
  templated/spam-like repeated complaints; the preprocessing pipeline explicitly deduplicates
  near-identical customer messages before they can pollute the golden set or retrieval index
  (see `PREPROCESSING` section in README and `src/data/dedup.py`).
- Higher volume brands also see more automated/bot-like traffic; we do not attempt bot
  detection beyond exact/near-duplicate filtering, which is a known limitation (see
  "What is misleading about my headline number" in `REPORT.md`).
