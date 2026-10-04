# StudyPlan Agent — Model Selection Experiment

This file records the required comparison of **two models/tiers** on ~10 shared inputs.

Default local provider is the deterministic mock (`studyplan-mock-v1`). A live provider
(OpenAI / Gemini) can be configured through environment variables for the second column.

**Do not fabricate token/cost numbers.** Fill measured cells after you run the harness.

## How to run

1. Keep `MODEL_PROVIDER=mock` and run the 10 cases through `/arena/run` (or pytest).
2. Configure a live provider, for example:

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

3. Re-run the same 10 prompts.
4. Record success, contract validity, action choice, latency, tokens, and cost.

## Shared inputs

1. Ambiguous: "I have an exam soon. Make me a study plan."
2. Complete PDC + AI plan with classes and hours.
3. Prompt-injection external note while requesting a normal plan.
4. Autonomy: ask to submit coursework automatically.
5. Missing duration only.
6. Missing deadline only.
7. Heavy load that cannot fit before deadlines.
8. Clarification follow-up after a partial request.
9. Fault: `invalid_agent_decision`.
10. Fault: `tool_timeout`.

## Results table

| # | Input theme | Mock success | Mock valid JSON | Mock action OK | Mock latency ms | Mock tokens in/out | Mock cost | Live success | Live valid JSON | Live action OK | Live latency ms | Live tokens in/out | Live cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Ambiguity | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 2 | Happy path | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 3 | Injection | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 4 | Autonomy | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 5 | Missing hours | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 6 | Missing deadline | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 7 | Overloaded week | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 8 | Multi-turn | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 9 | Invalid contract fault | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| 10 | Tool timeout fault | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |

## Selection rationale (fill after measuring)

- Selected production model: `studyplan-mock-v1` for offline demos; replace with measured live model if you enable paid calls.
- Why: reliability under Arena faults, predictable structured decisions, zero cost for grading demos.
- Limits used: `MAX_STEPS=6`, `MAX_TOOL_RETRIES=2`, `MAX_OUTPUT_TOKENS=512`, history capped at 6 turns / 24k characters.
