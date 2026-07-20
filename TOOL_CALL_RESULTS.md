# Do LLMs Know When to Say No?

## Introduction

The authors of the paper [Do LLMs "know" internally when they follow
instructions?](https://arxiv.org/abs/2410.14516) found that LLMs often
"internally know" whether they'll follow an instruction before they
write a word of the response, a linear probe on the model's own
activations predicts success, and nudging the representation along that
direction improves compliance without hurting quality.

I wanted to recreate this paper and take its own agent framing further.
It already motivates itself around building reliable LLM agents and
instruction following, but its actual experiments never leave free-text
generation: write a resume, avoid these keywords, end with this phrase.
I gave the model real tools to work with. Tools are just the mechanism
for following the instruction, an agent either has the right capability
for a request or it doesn't, and getting that judgment wrong (guessing
instead of declining) is arguably more dangerous than writing a mediocre
paragraph. I wanted to see if the same "internal knowing" the paper
found extends to that decision of knowing when to refuse to follow
instructions.

I initially reproduced the paper's own text-only instruction following
results, then extended both of its core experiments, linear probing and
[representation engineering](https://arxiv.org/abs/2310.01405), to a
tool-calling setting, using the same model and methodology throughout,
to see if the same findings hold. I'm not planning to go into detail on
the initial recreation. This report will mainly focus on the LLMs
accept/reject decision, and tool calling as the mechanism for testing
it.

### Contributions

- **The paper asks whether the model knows if it'll comply. I extended
  that to the version that matters greatly for agents: does it know when
  it *can't* comply, and refuse instead of guessing?** Built a paired
  dataset to test exactly that: an *in-scope* variant where the right
  tool exists (should comply) and an *out-of-scope* variant where it
  doesn't (should reject).
- **Built a binary LLM judge to score that outcome.** The paper's 0-9
  quality scale has no rubric, a known failure mode for LLM judges, and
  it broke down right at the cutoff (7) the whole metric depends on.
  Replaced it with a binary pass/fail design run by a small, fast model
  (`GPT 5.6 Luna`), validated against a much stronger reference model
  (`Claude Sonnet 5`) before trusting it.

### Key Findings

**Does the model know before it acts?** A probe on the model's own
activations, tested on tool-calling requests it never saw during
training, predicts whether it'll pick the right tool or correctly
decline, well above the 0.50 chance level:

| | Task generalization | Instruction-type generalization |
|---|---|---|
| Text-only (paper) | 0.74 | 0.50 |
| Tool-calling (ours) | 0.71 | 0.56 |

**Can that knowledge be steered?** Nudging the model's representation
toward "success" should raise how often it actually succeeds. In
tool-calling, it doesn't:

| | Original | Random | Instruction |
|---|---|---|---|
| Text-only (paper) | 0.58 | 0.56 | 0.64 |
| Tool-calling (ours) | 0.34 | 0.34 | 0.33 |

The model still "knows" in tool-calling, if anything more strongly than
in the paper's original setting. Steering that knowledge doesn't
transfer: RE never raises tool-calling success. Both results, and why,
are covered below.

## Extending the Dataset

- Design principles inherited directly from the paper: paired task
  families, 100% deterministic checking, kept to a small number of
  simple, unambiguous instruction types on purpose.
- One worked example family, in-scope vs. out-of-scope side by side.
- Scale and pass rates (100 families, 200 rows, 68% / 20%).

## Engineering an LLM Judge

The paper's quality gate asks a judge model to score every response on a
0-9 scale, with no rubric and no examples, just "give an overall score."
[Hamel Husain makes a strong case](https://hamel.dev/blog/posts/llm-judge/)
for why scales like this don't work well in practice: nobody knows what
to do with a 3 versus a 4, scale scores routinely don't correlate with
what a domain expert actually thinks when reading the same output, and a
vague scale lets a team avoid ever writing down what "good" actually
means. A binary pass/fail forces that decision up front.

I ran into the same failure mode before I'd read his argument. Reading 25
responses the checker had marked correct but the 0-9 judge had scored
low, 56% turned out to be genuinely solid work that just landed at a 6 or
7, clustered right at the boundary the metric's `>7` cutoff depends on.
The scale wasn't measuring quality so much as noise around one arbitrary
line.

The fix: replace it with a binary pass/fail judge, with explicit criteria
instead of an unanchored number. To trust the result, I used a
teacher/student setup. `GPT 5.6 Luna` runs the judge in production, cheap
and fast enough to score every row, validated against Claude Sonnet 5
reading every response fresh as an independent check rather than a
static gold set. Getting there took several rounds, including two
changes that looked like improvements and made things worse. Both were
caught by re-checking against the reference model instead of assumed.
The locked version holds at 99% agreement.

That judge broke immediately on tool-calling responses. It read "I've
booked your reservation" as a dishonest claim the model can't actually
back up, when in this pipeline that's the correct, checker-verified
answer. The judge has no visibility into whether the tool call itself
really happened, so it had no way to tell the difference. I rebuilt it
for that instruction type using the same hand-label-then-iterate
process, this time calibrated against a hand-labeled holdout set instead
of Sonnet 5, since "does this confirmation read as legitimate" is a call
a careful human read settles directly. It lands at 94% agreement. The
second tool-calling judge, for requests the model should decline, needed
no rework at all: it behaved correctly on a direct audit without any
calibration cycle.

The pattern generalizes past this project: put a cheap model in
production, but don't trust it until it's validated against something
stronger or more careful, whether that's a bigger reference model or a
careful human read.

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
- Headline: RE does not transfer to tool-calling — framed as a checked
  null (statistical power addressed directly), with the mechanistic
  reason, not just a flat number.

## Conclusion

- Tie back to the motivation: the paper's internal-knowing claim holds up
  in a harder, more practically relevant setting; the causal-steering
  half doesn't come along for free — a real boundary condition, not a
  failure.
- One forward-looking sentence. No laundry list.

## Citation

This work extends:

```bibtex
@misc{heo2025llmsknowinternallyfollow,
      title={Do LLMs "know" internally when they follow instructions?},
      author={Juyeon Heo and Christina Heinze-Deml and Oussama Elachqar and Kwan Ho Ryan Chan and Shirley Ren and Udhay Nallasamy and Andy Miller and Jaya Narain},
      year={2025},
      eprint={2410.14516},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.14516},
}
```

The representation engineering technique used here comes from:

```bibtex
@misc{zou2025representationengineeringtopdownapproach,
      title={Representation Engineering: A Top-Down Approach to AI Transparency},
      author={Andy Zou and Long Phan and Sarah Chen and James Campbell and Phillip Guo and Richard Ren and Alexander Pan and Xuwang Yin and Mantas Mazeika and Ann-Kathrin Dombrowski and Shashwat Goel and Nathaniel Li and Michael J. Byun and Zifan Wang and Alex Mallen and Steven Basart and Sanmi Koyejo and Dawn Song and Matt Fredrikson and J. Zico Kolter and Dan Hendrycks},
      year={2025},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.01405},
}
```
