# Judge audit: baseline run

**Why:** an LLM judge is only useful if its grades can be trusted. A second reviewer (Claude, an
AI assistant) re-graded a random sample of 30 of the 95 judged answers from
`answers_baseline.jsonl` (`random.seed(11)`), reading each answer next to the full text of the
sources it cited and the reference answer.

**Judge:** `groq:qwen/qwen3.8-27b`. **Answers:** `gemini:gemini-3.5-flash-lite` (section chunks,
hybrid search, top 5).

| | Agreement with the judge |
|---|---|
| Faithfulness (claims supported by cited sources) | **30 / 30** |
| Correctness (vs reference answer) | **29 / 30** |

**The one disagreement.** `s113` ("maximum duration for long duration crops under KCC"): the
answer said crop seasons are standardised at 18 months, which answers the question; the judge
marked it `partial` because the lower bound (more than 12 months) was missing. The judge leans
strict, so reported correctness is, if anything, slightly conservative.

**Claims checked beyond the first 900 characters of a source** (searched in the full chunk
text): `x004` (Table 1 caps ₹10,000 / ₹25,000), `s062` (ALCO transfer pricing policy), `s098`
(no movement of funds; downloadable forms), `s119` (half-yearly VA; new IT infrastructure),
`s031` (step-down subsidiary definition), `s097` (GAICA report, CERT-In auditors). All found.

Sample ids: s084 s102 s086 x004 s094 s106 s039 s037 m015 s087 s113 s110 s122 s018 x001 s062
s028 s017 s098 s007 s075 s119 s031 s002 s097 s012 s010 s006 s123 s049

**Limits of this audit.** The second reviewer is also an AI, not a domain expert; 30 is a small
sample; and the judge sees only the sources an answer cited, so "faithful" means "supported by
what it cited", not "complete".
