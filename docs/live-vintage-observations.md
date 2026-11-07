# Live vintage observations

## 2026-10-19: two Yahoo chart batches 14 minutes apart

Both batches requested the same inputs: `period1 = 2018-01-01`, `end_exclusive = 2026-10-17`, and all 24 ETFs.

**Batches:**

| Batch | Result | Rows per ticker |
|---|---|---|
| `20261019T072131358681Z` | 24 success | 2,210 |
| `20261019T073514451740Z` | 24 success | 2,210 |

**Bundles:**

- `20261016T20261019T072227Z-9075049d9cb8` (instant 07:22:27Z)
- `20261016T20261019T073556Z-533c04d9231a` (instant 07:35:56Z)

The target session is 2026-10-16 in both bundles, and both pass `verify`.

### What changed between the two batches (from `etf-monitor compare` and a direct Parquet diff)

- **Unchanged fields:** open, high, low, close and volume are identical in all 53,040 overlapping rows.
- **Adjusted close:** `adjusted_close` differs in 31,910 rows (60%).
  - Size of the change: max relative change 1.27e-6, max absolute change 3.7e-4, between 1 and 19 float32 ulps (median 1 ulp).
  - Both vintages are exactly float32-representable.
  - The changes are concentrated in older history. The last 51 sessions, including 2026-10-12 through 2026-10-16, are unchanged.
  - Yahoo appears to recompute and re-quantize its adjustment factors per request. **This explanation is an inference, not documented provider behavior.**
- **Retrieval timestamp:** `retrieved_at` changed in every row, as expected.
- **Signal changes across the full history:**
  - Momentum: 37,408 raw values changed and 8 percentiles changed.
  - Volatility: 45,301 raw values changed and 118 percentiles changed.
  - The percentile changes are rank-order flips between near-equal raw values.
- **Signal changes at the target session:**
  - Momentum raw/display values changed for QQQ, SPY, VGT, XLI and XLK. Their 2025-10-15 start endpoint was re-quantized, and the display return moved by at most 1.4e-5 percentage points.
  - No target-session percentile, status or reference count changed.

### Implications

- **Hashes:** Bundle content hashes are not stable across re-fetches, even when nothing economically meaningful changed. Reproducibility is therefore guaranteed per stored bundle, not per re-fetch. This is consistent with the design: bundles embed their input prices.
- **Tie rule:** Exact-equality ties (D6) are sensitive to this jitter in historical percentiles. The methodology was **not** changed. Whether to compare `adjusted_close` at float32 resolution, or to apply a tolerance, is an **unresolved methodological decision** for the owner.
- **Revision vs. jitter:** A vintage comparison cannot distinguish a genuine corporate-action revision from quantization jitter by counts alone. Magnitudes must be inspected.
