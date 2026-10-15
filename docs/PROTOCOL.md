# Preregistered research protocol

Primary classification comparison: deduplicated all-agree PhraseBank test, macro-F1,
three fixed labels; bootstrap stratified by truth, paired exact McNemar, no tuning on test.
Select API prompting only on validation macro-F1, tie break by name. Fixed TF-IDF C=1.
Secondary-only PhraseBank supplement is evaluated without training; it is not an independent
replication of the primary dataset. Twitter published validation is the OOD holdout.
FinancialPhraseBank and FinBERT overlap; FinBERT is a contaminated reference baseline.

Primary return hypothesis: positive minus negative historical sentiment buckets predict
market-adjusted returns over sessions +1 through +5. Entry in the tradable portfolio is at
the first scheduled open strictly after publication, hold five sessions. Day-zero prior-close
return is descriptive only. Daily cohorts are 20% of capital; each side is 50% of cohort
capital; absent legs remain cash. No forward filling, leverage expansion or future bucket edges.
Transaction cost: traded notional on both entry and exit, sensitivity 0/10/25 bps per side.
Fixed-capital cohorts imply changing gross exposure as equity evolves; report this limitation.
No borrow fees, locate constraints or open-auction slippage beyond flat costs are modeled.

Report all models, horizons and declared subsets. Best observed Sharpe is exploratory,
not a holdout-selected investment strategy. IC is monthly Spearman with minimum 10 events,
mean t-stat uses HAC(3). Event means cluster by event date; this does not fully handle firm
serial correlation or overlapping event windows. Report these limits and missing-price coverage.

Compare models on intersection of scored filing accessions when API is subsampled.
Historical market-cap strata require historical caps; absent caps mean unavailable, never
use present-day caps to label historical firms. Universe CSV must carry an as-of date and
source; current constituents carry survivorship bias, incomplete/delisted prices add selection.
Corporate identity/ticker changes need a reviewed CIK-to-security history before strong claims.
Yahoo adjusted closes are a free proxy, not a delisting-return-aware institutional database.

Pretraining may embed later news/returns. Only verified training cutoffs enable clean-period
subsets. Null cutoffs cannot support clean-period claims. Task training on old PhraseBank
sentences does not remove base-model contamination. Research use only; no investment advice.

Checkpoints: review benchmark with Nish before EDGAR scale-up. Stop on API budget breach,
OOD F1 >0.98 or Sharpe >2; write an audit (splits, duplicates, prices, entries, contamination)
before removing results/REVIEW_REQUIRED.json. Never report flagged output as a headline.
