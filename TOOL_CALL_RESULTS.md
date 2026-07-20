# Do LLMs Know When They Can Use a Tool?

*Draft outline — content in progress.*

## Introduction

*Self-contained — should stand alone even if no one reads past this section.*

- Reassert the paper's core claim in 2-3 sentences: LLMs encode a linear
  direction predicting instruction-following success, readable before
  generation starts, and steerable via representation engineering (RE).
- Pivot to tool calling: agents need both the ability to use a tool
  correctly *and* the judgment to decline when no tool fits. The paper
  never tested either. Frame this as the same two questions (does the
  model know it'll succeed? can that knowledge be steered?), asked in a
  harder, more agent-relevant setting.

### Contributions

- Extended the dataset to a tool-calling setting: paired task families,
  two instruction types — the agent has the right tool and should call it
  (in-scope), or has no tool that fits and should decline (out-of-scope).
- Reused the paper's probing and RE methodology unchanged, on the same
  model (Mistral-7B-Instruct-v0.3), for direct comparability against the
  paper's own numbers.
- Redesigned the quality judge from a 0-9 ordinal scale to a binary
  pass/fail rubric, grounded in a known LLM-judge failure mode and
  validated empirically.

### Key Findings

*Headline tables only — the whole story in miniature.*

- Table: probing AUROC, text-only (paper) vs. tool-calling (ours).
- Table: RE success rate (original / random / instruction-direction),
  text-only (paper) vs. tool-calling (ours).
- 2-3 sentence takeaway: the "knowing" signal replicates and even
  strengthens in tool-calling; the causal-steering result does not
  transfer — a real, evidenced dissociation, not a failed reproduction.

## Extending the Dataset

- Design principles inherited directly from the paper: paired task
  families, 100% deterministic checking, kept to a small number of
  simple, unambiguous instruction types on purpose.
- One worked example family, in-scope vs. out-of-scope side by side.
- Scale and pass rates (100 families, 200 rows, 68% / 20%).

## Linear Probes

- One-line method recap — identical to the paper's (same model, layers,
  token positions).
- Comparison table repeated/expanded from Key Findings.
- The nuance: instruction-type generalization mostly replicates the
  paper's at-chance result at the most comparable cell (first token,
  prompt-only) and only diverges at token positions that mean something
  structurally different in a short tool-call response vs. free text.

## Representation Engineering

- One-line method recap — identical hook mechanics to the paper's.
- Judge redesign as its own beat: the 0-9 scale's boundary problem, the
  binary fix, agreement numbers backing it up.
- Headline: RE does not transfer to tool-calling — framed as a checked
  null (statistical power addressed directly), with the mechanistic
  reason, not just a flat number.

## Conclusion

- Tie back to the motivation: the paper's internal-knowing claim holds up
  in a harder, more practically relevant setting; the causal-steering
  half doesn't come along for free — a real boundary condition, not a
  failure.
- One forward-looking sentence. No laundry list.
