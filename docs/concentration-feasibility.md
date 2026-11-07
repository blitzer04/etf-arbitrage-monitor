# Gate 7 — Concentration feasibility report

Status: **FAIL for a 24/24 daily signal — Concentration remains deferred, not calculated.** Research only. No sign-ups, no Concentration code.
Date of review: 2026-10-19.

## Findings

- **Legacy audit (2026-08-24).** 22/24 ETFs have daily complete holdings: 20 under Rule 6c-11, plus the SPY and DIA UITs via State Street daily files. VTI and VGT complete holdings are monthly, with about a 15-day lag.
- **2026-10-19 recheck of Vanguard.**
  - The Vanguard portfolio-holdings disclaimer says Vanguard may publish daily holdings for "certain Vanguard index ETFs".
  - However, the SAI text says index funds *not relying on Rule 6c-11* disclose month-end complete holdings 15 calendar days after month end.
  - VTI and VGT are ETF share classes of mutual funds, not Rule 6c-11 ETFs. A third-party guide (2026-07-12) also describes Vanguard full holdings as monthly.
  - The daily Vanguard file is a Portfolio Composition File. It is a creation-unit basket (share counts only, and it requires accepting terms), not complete holdings.
  - **The legacy finding stands.**
- **Entity mapping.** Mapping security to issuer to ultimate parent needs an official or licensed identifier source (OpenFIGI/GLEIF candidates); none has been validated. Cross-issuer schemas (derivatives, cash, negative weights, double listings) have not been reconciled.
- **No backward projection.** Today's holdings must not be projected backward to build historical concentration values.

## Cohort / time-policy alternatives (each needs owner approval; no silent switch)

| Option | Definition | Consequence |
|---|---|---|
| A | 24-fund common month-end cohort, published after the Vanguard lag (about T+15 calendar days) | Monthly signal with known availability dates. A 756-session-style reference window would need redefining in months, which is a methodology change. |
| B | Daily cohort of 22 funds (excluding VTI and VGT) | Changes the approved universe. Not approved. |
| C | Daily 24/24 | Not available from public sources today. |
| D | Daily PCF baskets as a proxy for VTI/VGT | A different object (creation basket, not holdings). Not equivalent; rejected unless explicitly approved as a proxy. |

## Recommendation

Keep Concentration deferred. To make progress, the owner would choose between Option A and Option B. Either path then needs:

- a validated entity-mapping source;
- a per-issuer schema reconciliation sample.

This would be done before any code.
