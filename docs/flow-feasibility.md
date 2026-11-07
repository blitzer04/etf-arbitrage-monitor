# Gate 6 — Flow feasibility report

Status: **FAIL — Flow remains deferred, not calculated.** Research only. No sign-ups, no paid trials, no Flow code.
Date of review: 2026-10-19. Sources: public web pages and documents only.

## What Flow would need

The spec defines a creation/redemption flow proxy from daily shares outstanding. Normalizing it the same way as the price signals needs:

- a daily shares-outstanding series for all 24 ETFs;
- publication timing, so each value can be dated to when it was observable;
- at least 252 eligible prior intervals per ETF, under the same midrank rule;
- licensing that permits local research use.

The spec also forbids stitching current-only snapshots into a history or backdating a first signal.

## Candidates

| Candidate | History | Timing / vintages | 24/24 coverage | Rights | Verdict |
|---|---|---|---|---|---|
| Yahoo/yfinance shares history (legacy audit 2026-08-21) | Ends 2021-03 | No vintages | 0/24 reach 252 intervals; 7 empty; QQQ/XOP same-date conflicts | Undocumented endpoint | FAIL |
| Issuer websites (current shares outstanding) | Current value only | Prospective capture only | Likely 24/24, unverified | Issuer terms vary | FAIL for history. A prospective collector would need about 13 months before the first normalized value. |
| SEC Form N-PORT public data sets | Oct 2019 to Sep 2026 | Monthly at best; only the quarter-end month report is public, released within 60 days of quarter end | Includes UITs (SPY, DIA) | Public | FAIL for a daily signal. At most a low-frequency cross-check. |
| DTCC ETF Portfolio Data Service / historical PCF | Historical PCF from 2007-11-01 (DTCC Learning) | NSCC processing date/time in file header | "Total Shares Outstanding Per ETF" is an **optional** PCF field (NSCC Notice 6660, 2008), so per-ETF coverage is unknown | Paid subscription; terms not public | UNRESOLVED. This is the only plausible path, but it needs licensing plus a measured sample. |
| Commercial ETF datasets (e.g. SQXray, FocusAlpha) | Vendor-claimed | Vendor-claimed | Unverified | Paid / sign-up | NOT EVALUATED (sign-up required) |

## Recommendation

Keep Flow deferred. To change that, the owner would have to approve one of these:

- **(a) DTCC historical PCF.** License it, then measure on a sample:
  - per-ETF share coverage of the optional field;
  - eligible interval counts;
  - duplicate/correction handling (the spec says "second is kept");
  - whether processing timestamps are retained in the history file.
- **(b) Prospective issuer-website capture.** Collect daily from now on, with stored raw files and capture timestamps. It produces no normalized values for about 13 months, and nothing can be backdated.

Neither option is implemented.

## Addendum (Gate 8, 2026-11-03) — partial PASS for SSGA funds

The issuer-website finding above was incomplete. State Street Global Advisors
publishes a free daily NAV-history workbook per SPDR ETF with NAV, shares
outstanding, and total net assets. On 2026-11-03 the file downloaded for all
25 SPDR funds tried (13 crowding-universe ETFs plus 12 arbitrage-universe
ETFs). From 2018-01-02 every file had one row per XNYS session, no
duplicates, and no missing shares outstanding.

Consequences:

- A **creation/redemption flow proxy** (daily change in shares outstanding ×
  NAV) is now feasible for those 25 funds. It is a separate Gate 8 output;
  the crowding signals and the deferred Flow signal in the methodology
  lockfile are unchanged.
- The other 11 crowding ETFs (Invesco, iShares, Vanguard, VanEck, ARK,
  Global X) remain without a machine-readable daily history; see
  `src/etf_monitor/config/issuer_nav_sources.yaml` for each reason. These are
  results of a limited scripted survey, not proof that no source exists.
- The history is the issuer's current file, not point-in-time vintages.
  Restatements are only detectable going forward by comparing NAV vintages.

- Live check (2026-11-04 vintages): SSGA does not restate NAV or shares
  outstanding for splits (XLB/XLE/XLK/XLY 2:1 on 2025-12-05, XOP 1:4 on
  2020-03-30, JNK 1:3 on 2019-05-06). Premium uses the as-traded close and
  split sessions carry no flow proxy; see the decision log.
