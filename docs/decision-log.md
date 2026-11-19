Owner approved all proposed defaults D1–D12 on 2026-10-19.

## 4. Methodology lockfile — APPROVED (carried verbatim from spec §3; changes need your signoff)

- Universe: exactly the 24 ETFs in spec §2, fixed category mapping, versioned + hashed.
- Calendar: XNYS regular sessions, exchange-calendars pinned; early-close day = one session; holidays not sessions.
- Price input: canonical finite positive `adjusted_close` only; never `close` substitution.
- Momentum: `R_t = ln(A_t/A_(t-252))` on exact XNYS positions, no skip; display `100*(A_t/A_(t-252)-1)`; missing exact endpoint -> NaN; interior gaps are diagnostics only.
- Volatility: 22 consecutive positions -> 21 adjacent log returns, `ddof=1`, `* sqrt(252)`; any missing row/price in chain -> NaN; constant returns -> valid 0.
- Normalization (both): per ETF; reference = eligible prior raw values dated `d_{t-755}..d_{t-1}` (current excluded); N >= 252; `100*(#< + 0.5*#=)/N`; all-tied -> 50; no z-score/winsorize/clip/fill.
- Timing: labelled as of close of `d_t`; first prospective-use = next XNYS session.
- Missing target never rolls back; stale value never displayed as current.
- Flow, Concentration, Composite: deferred. No score, no label, no reweighting, no zero/neutral fill.
- Historical series labelled current-vintage reconstruction.

## 5. Resolved by owner (defaults adopted)

| # | Decision | Why it matters | Proposed default |
|---|---|---|---|
| D1 | New repo location/name | Spec forbids touching legacy repo; creating a GitHub repo is an external action. | **Build locally in `~/repos/etf-monitor` now; create private GitHub repo `blitzer04/etf-monitor` only when you approve (needed for CI + PR-per-gate).** |
| D2 | Yahoo transport | Private yfinance API is fragile across versions (1.5.2 -> 1.7.0); own HTTP client means owning cookie/crumb/rate-limit handling and is still an undocumented endpoint. | **(c) Minimal own HTTP client (stdlib/`requests`-free `urllib`) behind `transport.py`, parser fully offline-tested on recorded payload fixtures; no dependency on yfinance private internals.** If Yahoo requires crumb auth, fall back to (a) pinned yfinance with a contract test. |
| D3 | Price history start date | Spec cites "2018 onward" for Flow; legacy used 2018-01-01 inclusive. Determines first eligible percentile (session 505). | **2018-01-01 inclusive, end exclusive = day after target.** |
| D4 | Target-session rule | Not restated in new spec. Legacy: latest XNYS session whose scheduled close <= captured UTC instant, no grace; plus a 2h *operational* buffer before fetching. | **Adopt legacy rule unchanged; 2h buffer as an operational warning, not a methodological shift.** |
| D5 | Canonical numeric types | Spec requires lossless casts and rejection of precision loss; tests at 2^53, 2^53+1, 2^63+1. Legacy stored everything incl. volume as float64 (volume > 2^53 rejected). | **Prices float64 (JSON source is float); volume nullable Int64 (exact to 2^63-1; reject 2^63+1, negatives, non-integral). This is a schema change vs legacy — flagging for approval.** |
| D6 | Tie definition in midrank | Revisions produce float32-quantized near-ties. | **Exact float equality, no tolerance (legacy behaviour).** |
| D7 | Dependence diagnostics inputs and minimum pairs | Spec says "Pearson/Spearman with pair count and status", no minimum given. | **On normalized percentiles, exact-date inner join, per-ETF and per-session, min 3 pairs and >=2 distinct values per side; no p-values, no pooled estimate (legacy).** |
| D8 | Volume outlier flag (IWM 2019-07-30) | Spec wants anomalies flagged, but any threshold would be an invented rule; volume feeds no MVP signal. | **No automated volume-outlier rule in MVP. Record the known IWM observation in the failure register as "unverified anomaly"; volume shown raw with no derived liquidity metric.** |
| D9 | Live data in this build | DoD needs two distinct verified bundles. Each live run = one 24-ETF Yahoo batch. Legacy runbook requires per-batch authorization. | **Ask you to authorize up to two live 24-ETF batches (at least ~hours apart, after 2h post-close) to produce two real vintages; otherwise two clearly labelled FICTIONAL synthetic bundles.** |
| D10 | Revision policy on overlap | Spec: monotone provenance, reject same-timestamp conflicts and stale updates, snapshot before replace, fail on field loss / vanished in-coverage rows. | **Adopt exactly; accepted revisions counted per ticker and surfaced; no provider-cause inference.** |
| D11 | Non-session source dates | A Yahoo date that is not an XNYS session (e.g. holiday artefact). | **Reject the ticker response (no drop, no remap).** |
| D12 | Gates 6/7 scope in this session | Spec says research-only, no signups. | **Desk research from public docs only; deliver dated feasibility reports with pass/fail; Flow/Concentration stay deferred.** |

## Gate 3 — standalone signal calculations

- The in-memory panel uses every XNYS session from the methodology history
  start through the target; canonical rows outside that grid are rejected.
- Momentum uses exact adjusted-close endpoints 252 sessions apart. Interior
  gaps are counted as diagnostics and never shift the endpoints.
- Volatility uses 21 adjacent log returns from exactly 22 prices, sample
  standard deviation (`ddof=1`), and `sqrt(252)` annualization. A missing row or
  adjusted close invalidates only windows containing it; zero volatility is
  valid.
- Each raw series is normalized against eligible prior observations for the
  same ETF, with the current session excluded, a 756-session reference window,
  a 252-observation minimum, and exact-float midrank ties.
- Outputs are as of the signal-session close; prospective use begins on the
  next XNYS session. No fill, composite, crowding score, or label is produced.

## Gate 2 compatibility note

NumPy is constrained to `>=1.26,<2` because importing the approved
`exchange-calendars==4.13.2` under NumPy 2.x emits a deprecation warning, which
fails the repository's warnings-as-errors checks. Revisit this constraint
whenever the exchange-calendars pin changes; do not remove it without repeating
the import and warning checks. PyArrow is bounded below 20 so its locked wheel
remains compatible with the NumPy 1.x constraint; revisit that bound alongside
the NumPy constraint.

## Gate 5 — local viewer and transport observation

- One Yahoo request returned HTTP 429 with the full Chrome user-agent string
  and HTTP 200 with plain `Mozilla/5.0`; the transport uses the latter. This is
  a single observation, not a documented provider contract or availability
  guarantee.
- No candidate run is selected by default. Candidate directories remain
  unverified until the user explicitly selects one.
- Selection triggers package bundle verification. The viewer caches a
  successfully verified immutable bundle for the lifetime of the process.
- The app does not independently calculate signals or metrics; it delegates
  verification, including the required output recomputation, to the package
  verifier and displays the returned artifacts.

## Gate 8 — issuer NAV, premium/discount, creation/redemption flow proxy

Approved by the owner on 2026-11-03 ("go"). These additions are a separate
data path; the crowding universe, adjusted-close Momentum/Volatility, and
bundle formats are unchanged.

- **Source.** SSGA daily NAV-history workbooks
  (`navhist-us-en-<ticker>.xlsx`): NAV, shares outstanding, total net assets.
  The layout is an observation, not a documented contract; the parser
  (`providers/ssga_nav.py`) validates identity, header, date labels, order,
  and values and rejects the whole ticker on any violation.
- **Coverage.** 13 of the 24 crowding ETFs are SPDR funds with SSGA files.
  The other 11 are listed with a reason in `config/issuer_nav_sources.yaml`
  and reported as `unavailable`; nothing is estimated for them.
- **Arbitrage universe.** A separate fixed 12-ETF list
  (`config/arbitrage_universe.yaml`: SPDR fixed-income, international, and
  global funds) with its own hash, loader, and price store under
  `<data-root>/arbitrage/`. It never enters crowding evaluation.
- **History window.** Rows before `price_history_start` (2018-01-01) are
  counted but neither validated nor kept. Example: BIL publishes fractional
  shares outstanding in December 2017.
- **Missing values.** SSGA's `-` stays missing (`<NA>`/`NaN`). Missing XNYS
  sessions inside a ticker's range are listed in the manifest, never filled.
- **Persistence.** Each download is an immutable vintage directory with the
  exact raw workbooks, `nav.parquet`, and a manifest of statuses, UTC
  retrieval instants, and SHA-256 digests. Loading verifies every digest and
  re-derives all rows from the raw files. Vintages are not merged; issuer
  restatements show up as differences between vintages.
- **Raw close for premium only.** `premium_discount_bps = 10,000 × (close /
  NAV − 1)` uses Yahoo's raw `close` because NAV is the same session's
  per-share value; dividend back-adjustment would distort the comparison.
  `adjusted_close` remains the only input to Momentum and Volatility.
- **Flow proxy, both alignments.** `delta_shares` is the exact integer change
  from the previous XNYS session (never the last available row).
  `flow_proxy_same_session_usd = delta_shares × NAV[t]` and
  `flow_proxy_prior_session_usd = delta_shares × NAV[t−1]` are both kept
  until the lead-lag diagnostic shows which reporting alignment the data
  supports. Both are a **creation/redemption flow proxy**, not reported flows.
- **Live requests.** `fetch-issuer-nav` refuses to run without
  `--authorize-live-request`, matching `fetch-prices`.
- **Share splits (verified on live data).** Yahoo `close` is
  split-adjusted; SSGA does **not** restate historical NAV or shares
  outstanding. Before correction, the first live run showed XLB/XLE/XLK/XLY
  near -5,000 bps before their 2025-12-05 2:1 splits, XOP near +30,000 bps
  before its 2020-03-30 1:4 reverse split, and JNK near +20,000 bps before
  its 2019-05-06 1:3 reverse split, with share counts jumping by the split
  ratio on the split session. Rule: split events are read from the same raw
  Yahoo chart payloads that produced the prices (`--price-raw-dir`), the
  premium uses `close_as_traded = close * split_factor` (product of
  `numerator / denominator` over later splits), and the split session's
  share change is not a flow (`split_session`, null proxy). After
  correction every core fund's median premium is within 1 bp of zero.

## Gate 9: holdings-based basket value (2026-11-03)

- Source: SSGA `holdings-daily-us-en-<ticker>.xlsx`, same redirect and
  user-agent behavior as the NAV-history files. Only the latest snapshot is
  served (`Holdings: As of DD-Mon-YYYY`); there is no archive, so holdings
  history starts with the first capture and is never backfilled. On
  2026-11-04 07:08 UTC the snapshot was dated 2026-11-02 while NAV for
  2026-11-03 was already published, so the snapshot's date semantics are
  tested with two alignments (`same_session`, `next_session`) instead of
  assumed.
- Two layouts are accepted and anything else is rejected: equity (Shares
  Held) and fixed income (Par Value, Market Value, optional ISIN). Lines are
  kept in published order and not deduplicated (duplicate tickers and
  identifiers are real, e.g. two listings or currency balances).
- Independent valuation only where an independent price exists: U.S.-only
  equity funds at component Yahoo closes (dot share classes mapped to Yahoo
  hyphens, split-corrected to as-traded). Index futures are valued at zero
  (variation margin is in cash); rights, earn-outs and non-USD cash are
  excluded and reported as `unpriced_weight_pct`; no issuer weight is used to
  fill them. Funds with any non-USD security are not valued (no FX or
  foreign-close alignment). Fixed-income funds get only an issuer
  market-value consistency check, labelled `issuer_market_value`.
- A missing component close makes the basket null with
  `missing_component_price`; nothing is carried forward.

### Gate 9 timing: `As of D` holdings are the portfolio for session D+1 (2026-11-05)

Checked on the 13 core funds with the snapshot labelled 2026-11-02
(captured 2026-11-04 07:08 UTC), component closes through 2026-11-04 and an
NAV vintage captured 2026-11-05 20:25 UTC.

| Comparison | `same_session` (D) | `next_session` (D+1) |
| --- | --- | --- |
| `implied_shares_outstanding` vs reported shares outstanding | −2.84% to +6.73% | 12/13 within ±0.10%; XLP −0.51% |
| `basket_vs_nav_bps` | −284 (XOP) to +673 (KRE) | 12/13 from −9.6 (XBI) to +3.9 (DIA); XLP −51.6 |

- The share quantities in a file labelled D already include D's
  creations/redemptions, which SSGA reports in the D+1 shares outstanding.
  `next_session` is the alignment to use; `same_session` stays in the output
  as a diagnostic, and its gaps are timing artifacts, not dislocations.
- The two rows are one comparison in two units: `implied_shares_outstanding`
  is basket value / NAV, so its gap to reported shares equals
  `basket_vs_nav_bps` / 10,000. They are not independent evidence.
- Under `next_session`, 10 of the 12 funds other than XLP are negative, from
  −0.2 (XLK) to −9.6 (XBI); XOP (+1.1) and DIA (+3.9) are positive. XLP is
  −51.6 bps; its cause (share count, a component price, or omitted holdings)
  is not identified. The mostly negative residual is not explained yet.
  Accrued income and expenses not listed as holdings are a likely cause, but
  this is not verified.
- Second snapshot (checked 2026-11-06): the file labelled 2026-11-04
  (captured 2026-11-05 20:25 UTC), valued with component closes refetched
  2026-11-06 22:09 UTC and an NAV vintage captured 2026-11-06 22:08 UTC. A
  component vintage fetched at 2026-11-05 20:45 ET had null 2026-11-05 closes,
  so `next_session` correctly returned `missing_component_price` until the
  refetch. OKTA is covered. All 26 rows are `ok`.

| Comparison (snapshot labelled 2026-11-04) | `same_session` (D) | `next_session` (D+1) |
| --- | --- | --- |
| `implied_shares_outstanding` vs reported shares outstanding | −1.37% to +10.35% | 12/13 within ±0.10%; XLP −0.50% |
| `basket_vs_nav_bps` | −137 (XBI) to +1,035 (KRE) | 12/13 from −9.9 (XBI) to +3.9 (DIA); XLP −50.3 |

- The two snapshots agree: `next_session` holds for both. Of the 12 funds
  other than XLP, 8 are negative this time (SPY −2.2, XLB −3.3, XLI −3.9,
  KRE −4.0, XLF −7.0, XLE −7.4, XLV −7.6, XBI −9.9) and 4 are positive (XLY
  +0.0, XLK +0.4, XOP +1.2, DIA +3.9). DIA is +3.9 and XLP about −50 bps in
  both snapshots, so XLP's gap looks systematic rather than noise; its cause
  is still not identified. Two snapshots support the alignment rule but do not
  prove it for every future file; the private capture process keeps adding
  snapshots.

### Gate 9 follow-up: ten snapshots and ex-dividend drift (2026-11-18)

- Inputs: ten holdings snapshots labelled 2026-11-02 and 2026-11-04 through
  2026-11-16 (consecutive sessions from 11-04; the daily capture started on
  11-06, so 11-03 was never captured), each valued with the component vintage
  captured in the same run, and NAV vintage `nav-20261118T101802271179Z`.
  Reproduce with `basket-history` (`summary.json` records every input path).
- All 130 `next_session` fund-sessions are `ok`. |`basket_vs_nav_bps`| is at
  most 5 bps for 56, 10 bps for 97, 15 bps for 123, and 30 bps for 128. The
  two above 30 bps are XLP in the first two snapshots (−51.6 and −50.3).
- The "12 of 13 within 10 bps" result held for the first two snapshots only.
  Funds within 10 bps fall from 12 to 7 across the ten snapshots because gaps
  drift negative: XLE −7.3 to −25.6, XOP +1.5 to −26.0, KRE −2.6 to −16.6,
  XLB −3.3 to −14.7.
- Cause: ex-dividend receivables. On an ex-date the component close drops by
  the dividend, NAV books a dividend receivable, and the holdings file does not
  list it. `ex_dividend_bps` sums quantity × Yahoo dividend (split-corrected to
  as-traded) for ex-dates after the previous snapshot's valuation date and up
  to this one. For the 18 fund-intervals with `ex_dividend_bps` ≥ 2,
  `gap_change_bps + ex_dividend_bps` has median 0.18 bps (median absolute
  0.24 bps). Intervals with no ex-dates have median |`gap_change_bps`| 0.22 bps.
  Examples: XOP −8.9 against 9.0, XLB −6.7 against 6.8.
- Dividend cash lifts the gap back when it arrives. XLE on 2026-11-16: XOM and
  COP went ex (20.1 bps) but the gap moved only −13.3 bps; the cash line rose
  6.5 bps (`cash_change_bps`) the same day. Payment dates are not in the Yahoo
  payload, so the outstanding receivable balance, and hence the level of each
  gap, is not modelled; only interval changes are tested.
- Correction to the timing note above: XLP's −50 bps is not systematic. From
  the 2026-11-05 snapshot on, XLP is −6 to −14 bps like the other funds. The
  two early gaps look like separate one-off items (inference, not verified):
  in the 11-02 file every XLP quantity is 3.68% above the 11-05 file while
  reported shares outstanding are 4.12% above, so the holdings cover about
  0.4% fewer fund shares (about −42 bps); the 11-04 file has a −$47.2M USD
  cash line (−31 bps of net assets) and no money-market line.
- Capture timing: the cron is nominally 02:17 UTC, but GitHub started every
  scheduled run between 08:53 and 10:48 UTC. Each run therefore captured the
  holdings labelled D−1 together with the NAV and closes for session D, which
  is everything one `next_session` check needs. No label was missed.
- Implication for the band: the U.S. equity bands (about 3 bps) are smaller
  than this drift, so a `next_session` basket gap is not by itself evidence of
  a dislocation until receivables are accounted for.

## Gate 10: cost-based arbitrage band and AP response (2026-11-05)

**Band.** `band_bps = fee_bps + basket_cost_bps + half_spread_bps`, one-way, in
bps of NAV (`src/etf_monitor/arbitrage/band.py`). Inputs live in
`src/etf_monitor/config/arbitrage_costs.yaml` with a source per value:

- `fee_bps = 10_000 * creation_fee_usd / (creation_unit_shares * nav)`.
  Creation-unit size: Form N-CEN item E.2 from each trust's latest N-CEN on
  EDGAR. Fixed fee: the trust's SAI fee table (485BPOS); BWX, EBND, SPY and
  DIA are not in a table I located, so they use the N-CEN reported average
  fee per unit. Variable charges (up to 3x for cash/non-standard orders) are
  excluded. Fees are small: 0.2 bps (SPY) to about 2 bps (SPDW, EBND).
- `basket_cost_bps` and `half_spread_bps`: **assumptions** by asset class,
  not observations. They dominate the band.
- The Abdi-Ranaldo OHLC spread estimate (`spread.py`) was tried as the ETF
  spread input and rejected: the 21-session estimate is zero in 37-54% of
  windows and far above quoted spreads otherwise (SPY median 16 bps). It is
  kept in the output as `spread_estimate_bps` for reference only.

**Flags.** `side` is `premium`/`discount` when `|premium| > band`;
`persistent` requires the previous NAV row on the same side. `stale_nav`
funds (international) are flagged but reported separately: their premium
mixes mispricing with U.S. moves after the foreign close.

**AP response test** (`response.py`). For each session t,
`forward_rel_change = SO[t+5] / SO[t] - 1` (window starts at t+1 because SSGA
reports creations one session late). Share of sessions followed by net
creation after a premium, minus the same share for inside-band sessions;
likewise redemption after a discount. Percentage points, 95% interval from a
500-draw bootstrap resampling calendar months (`arbitrage-20261105-v2`,
`core-20261105-v2`, 2018-01 to 2026-11):

| Asset class | Creation after premium | Redemption after discount |
|---|---|---|
| U.S. high yield (JNK, SJNK) | +18.4 [13.1, 23.4] | +16.7 [8.1, 25.3] |
| U.S. investment grade (SPAB, SPSB) | +11.7 [6.1, 17.7] | +45.8 [14.0, 58.6] (43 events) |
| EM local bond (EBND) | +25.8 [19.2, 32.9] | +24.3 [13.8, 33.3] |
| Intl government bond (BWX) | +20.1 [14.5, 25.6] | +13.9 [6.3, 21.9] |
| EM equity (SPEM) | +16.1 [11.3, 20.8] | +5.3 [1.7, 8.7] |
| Developed intl equity (SPDW, FEZ, GII, GNR) | +7.6 [4.6, 10.8] | +8.9 [6.4, 11.4] |
| U.S. large-cap equity (SPY, DIA, XL*) | +3.7 [1.5, 6.0] | +4.5 [2.3, 6.9] |
| U.S. small/mid equity (KRE, XOP, XBI) | +2.2 [-1.7, 5.8] | -3.0 [-7.1, 1.2] |

Reading: APs respond to out-of-band premiums and discounts in bond funds;
U.S. equity funds are outside their ~3 bps band in 10-30% of sessions but
flows barely respond, so most of those flags are close/NAV measurement noise
rather than exploitable dislocations. Caveats: 5-session windows overlap;
the proxy cannot separate AP arbitrage from other primary-market demand;
the costs are assumptions, so flag counts scale with them.
