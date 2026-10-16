# DR-NNNN: <Decision in a short sentence>

| Field | Value |
|---|---|
| Status | Proposed \| Accepted \| Rejected \| Superseded by DR-NNNN \| Deprecated |
| Date | YYYY-MM-DD |
| Deciders | <names / roles> |
| Consulted | <who gave input> |
| Related | <links to issues, PRs, other DRs> |

## Table of Contents

- [Context](#context)
- [Decision drivers](#decision-drivers)
- [Options considered](#options-considered)
- [Decision matrix](#decision-matrix)
- [Decision](#decision)
- [Consequences](#consequences)
- [Follow-ups](#follow-ups)
- [References](#references)

## Context

<What is the problem? Why decide now? What constraints exist? 3–6 sentences.>

```
<optional: ascii diagram of the current state / the part of the system affected>
```

## Decision drivers

What matters most, in priority order. These are the columns in the matrix.

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | <e.g. Performance> | 3 |
| D2 | <e.g. Operational simplicity> | 2 |
| D3 | <e.g. Team familiarity> | 1 |

## Options considered

### Option A: <name>

<One-line description.>

- Pros:
  - ...
- Cons:
  - ...
- Cost / effort: <S / M / L>

### Option B: <name>

<One-line description.>

- Pros:
  - ...
- Cons:
  - ...
- Cost / effort: <S / M / L>

### Option C: Do nothing

<What happens if we don't change anything?>

## Decision matrix

Score each option 1 (poor) to 5 (great) per driver. Total = sum of score × weight.

| Option | D1 (×3) | D2 (×2) | D3 (×1) | Total |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |

The matrix informs the decision; it does not make it. Explain any override below.

## Decision

We choose **Option <X>** because <main reason, tied to the top drivers>.

```
<optional: ascii diagram of the target state>
```

## Consequences

| Type | Item |
|---|---|
| Good | ... |
| Bad | ... (what we accept / give up) |
| Risk | ... → mitigation: ... |

## Follow-ups

- [ ] <task> — owner — due date
- [ ] Revisit this DR if <trigger, e.g. load > 10k rps>

## References

- <links>
