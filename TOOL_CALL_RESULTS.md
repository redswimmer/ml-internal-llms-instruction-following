# Do LLMs Know When to Say No?

## Introduction

The authors of the paper [Do LLMs "know" internally when they follow
instructions?](https://arxiv.org/abs/2410.14516) found that LLMs often
"know" whether they'll follow an instruction before they
write a word of the response, a linear probe on the model's own
activations predicts success, and nudging the representation along that
direction improves compliance without hurting quality.

I wanted to recreate this paper and take its own agent framing further.
It already frames itself around building reliable LLM agents and
instruction following, but its actual experiments never leave free-text
generation: write a resume, avoid these keywords, end with this phrase.
I gave the model real tools to work with. Tools are just the mechanism
for following the instruction, an agent either has the right tool
for a request or it doesn't, and getting that judgment wrong (guessing
instead of declining) is arguably more dangerous than writing a mediocre
paragraph. I wanted to see if the same "internal knowing" the paper
found extends to that decision of knowing when to refuse to follow
instructions.

I initially reproduced the paper's own text-only instruction following
results, then extended both of its core experiments, linear probing and
[representation engineering](https://arxiv.org/abs/2310.01405), to a
tool-calling setting, using the same model and methodology throughout,
to see if the same findings hold. I ran everything on
Mistral-7B-Instruct-v0.3, the only one of the paper's four models with
real tool-calling support, on an RTX 4090 (24GB VRAM), and stuck with
their model rather than a newer one like Qwen to keep the comparison
direct. 

I'm not planning to go into detail on the initial text-only recreation of the paper and instead focus on the LLMs accept/reject decision, and
tool calling as the mechanism for testing it.

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
  Replaced it with a binary pass/fail design run by a small, fast
  student judge (`GPT 5.6 Luna`), which I iterativly aligned and validated against
  a much stronger auditor judge (`Claude Sonnet 5`) before promoting it to the production judge.

### Key Findings

#### Does the model know before it acts?
A probe on the model's own activations, tested on tool-calling requests
it never saw during training, predicts whether it'll pick the right tool
or correctly decline. That question splits into two different tests, and
they don't tell the same story. **Task generalization** -- an unseen
request, same instruction type the probe trained on -- is well above the
0.50 chance level across the board. **Instruction-type generalization**
-- an instruction type the probe has never seen a single example of --
is a much harder test and stays close to chance for text-only; it's
still low, just consistently a few points above chance for tool-calling:

| | Task generalization | Instruction-type generalization |
|---|---|---|
| Text-only (paper) | 0.74 ± 0.02 | 0.50 ± 0.05 |
| Text-only (ours) | 0.737 ± 0.037 | 0.538 ± 0.063 |
| Tool-calling (ours) | 0.706 ± 0.059 | 0.555 ± 0.006 |

![AUROC comparison: task generalization is well above chance for the paper and both of our runs. Instruction-type generalization is a much weaker signal throughout -- at chance for the paper, only a few points above chance for either of our own runs.](assets/fig1_auroc_headline.png)

Text-only (ours) is my own recreation of the paper's experiment: same
model, same method, run on my own hardware with my own seeds. It lands
close to their published numbers, not exactly on them. Tool-calling (ours) runs that identical pipeline,

#### Can that knowledge be steered?
In the paper, nudging the model's representation along what it calls the
"instruction-following direction" raised the success rate, the fraction
of responses that both pass the deterministic checker and clear a
quality bar. To check the effect is coming from that specific direction
and not just from perturbing the activation at all, the paper also
compares against nudging by the same amount in a random direction. I
found the instruction-following direction doesn't help, in tool-calling.

| | Original | Random | Instruction-follow |
|---|---|---|---|
| Text-only (paper) | 0.58 ± 0.00 | 0.56 ± 0.02 | 0.64 ± 0.02 |
| Text-only (ours) | 0.525 ± 0.00 | 0.533 ± 0.004 | 0.570 ± 0.00 |
| Tool-calling (ours) | 0.338 ± 0.00 | 0.340 ± 0.003 | 0.325 ± 0.00 |

![Success rate across Original, Random, and Instruction-follow conditions, for the paper, our text-only recreation, and our tool-calling extension. Paper and text-only both climb left to right; tool-calling is flat-to-declining and ends up lowest at Instruction-follow.](assets/fig1b_re_headline_sr.png)

Original and Instruction-follow are truly deterministic on my side, greedy
decoding against a fixed direction, so ± 0.00 is exact, not just one
run. The paper's own Instruction-follow std comes from retraining their probe
per seed; my direction is a closed-form mean-difference vector instead
(why, in Representation Engineering below), so there's no seed variance
left to average there.

The model still "knows" in tool-calling. If anything it knows more
strongly than in the paper's original setting. But steering that
knowledge doesn't transfer. The instruction direction drops below both
original and random, where in the text-only setting it clearly beat
both. Both results, and why, are covered below.

## Extending the Dataset

I built this [dataset](data/tool_calling.jsonl) the same way the paper builds IFEval-simple: pair
every instruction condition against the same set of tasks, so a probe's
signal can be attributed to the instruction, not incidental task
content. Here's how the two line up:

| | Paper (IFEval-simple) | Tool-Calling (ours) |
|---|---|---|
| Task, held fixed per pair | request content, e.g. "write a resume" | request content, e.g. "reserve a hotel room for Jordan Lee" |
| What varies | textual constraint, e.g. "avoid these keywords" | tool menu: whether a tool exists to fulfill the request |
| Scoring | deterministic checker | deterministic checker |

In practice: both rows in a pair ask for the exact same thing, word for
word. What changes is only the tools on offer. If one of them can
fulfill the request, the model should call it. If none can, it should
reject. Here's what that looks like for one real task,
`hotel_booking_000`, straight from `data/tool_calling.jsonl`:

**Should follow the instruction** (`tool:in_scope` in the dataset). A
tool exists (`reserve_hotel_room`), so the correct move is to call it:

```json
{
  "prompt": "Please reserve a standard king room at the Harborview Grand Hotel for Jordan Lee, checking in on July 20, 2026 and checking out on July 23, 2026; I need a regular hotel room, not a suite.",
  "tools": ["reserve_hotel_room", "reject", "reserve_hotel_suite"],
  "correct_tool_name": "reserve_hotel_room",
  "correct_args": {
    "hotel_name": "Harborview Grand Hotel",
    "room_type": "standard king room",
    "check_in_date": "July 20, 2026",
    "check_out_date": "July 23, 2026",
    "guest_name": "Jordan Lee"
  }
}
```

**Should reject** (`tool:out_of_scope` in the dataset). Same prompt, but
this time none of the offered tools can fulfill it
(`book_notary_appointment` and `create_calendar_event` are real tools,
just unrelated to a hotel booking), so the correct move is `reject`:

```json
{
  "prompt": "Please reserve a standard king room at the Harborview Grand Hotel for Jordan Lee, checking in on July 20, 2026 and checking out on July 23, 2026; I need a regular hotel room, not a suite.",
  "tools": ["reject", "book_notary_appointment", "create_calendar_event"],
  "correct_tool_name": "reject",
  "correct_args": {}
}
```

Note: Trimmed to the fields that matter here; each row's full JSON also
carries the JSON-schema definition for every listed tool, since that's
what actually gets passed to the model.

This dataset backs two separate checks on every generated response: a
deterministic checker, and an LLM judge.

**Deterministic checker.** The model generates a response from the row's
prompt and tools. The checker parses out the tool call it made and
compares it to `correct_tool_name` and `correct_args`. Exact match on
both is a pass, anything else a fail. This is the only thing deciding
whether a response is instruction-following-correct.

**LLM judge.** The same response is also scored for quality, separately,
since a technically correct tool call can still read badly to a user.
The dataset has no ground truth for that, it's judged fresh each time.

The checker's pass/fail is what the linear probes below are trained to
predict. The judge only enters later, as part of representation
engineering's quality gate.

## Engineering an LLM Judge

The paper's quality gate asks a judge model to score every response on a
0-9 scale, with no rubric and no examples, just "give an overall score."
[Hamel Husain makes a strong case](https://hamel.dev/blog/posts/llm-judge/)
for why scales like this don't work well in practice: nobody knows what
to do with a 3 versus a 4, scale scores routinely don't correlate with
what a domain expert actually thinks when reading the same output, and a
vague scale lets a team avoid ever writing down what "good" actually
means. A binary pass/fail forces that decision up front.

### Failure Mode
I ran into the same failure mode with the paper's LLM judge (GPT-4). While performign error analysis and reading responses the checker had marked correct but the 0-9 judge had scored
low, 56% turned out to be genuinely solid work that just landed at a 6 or
7, clustered right at the boundary the metric's `>7` cutoff depends on.
The scale wasn't measuring quality so much as noise around one arbitrary
line. Here's the actual prompt producing that scale, unchanged from the
paper:

```
You are a helpful assistant in evaluating the quality of the outputs for a given instruction.
Your goal is to score a given output for the given instruction. You should give an overall score
(an integer) on a scale of 0 to 9, where a higher score indicates better overall performance.
Do NOT provide any explanation for your evaluation.

# Instruction: {Task-only-input}
# Output: {Response}
# Score of the Output (Your response should be ONLY the score, an integer between 0-9):
```

### The Fix
I replaced it with a binary pass/fail judge, with explicit criteria
instead of an unanchored number. To trust the result, I used a
student/auditor setup: `GPT 5.6 Luna` is the student judge, cheap and
fast enough to score every row in production, and `Claude Sonnet 5` is
the auditor, reading every response fresh as an independent check. Getting to 94%+ agreement took several iterative rounds of prompt engineering with the auditor critiquing the student judge's output.

**The loop:**

```mermaid
flowchart TD
    A[Write/revise binary judge criteria] --> B["Score every response with<br/>GPT 5.6 Luna (student judge)"]
    B --> C["Independently re-score the same<br/>responses with Claude Sonnet 5 (auditor judge)"]
    C --> D{Do the two align?}
    D -->|No, revise| A
    D -->|Yes| E[Promote Luna to production judge]
```

### LLM Judge Roles

Each judge has its own job, scored against explicit pass/fail criteria
instead of a vibe: the comply judge checks the model's confirmation, the
decline judge checks its refusal. Neither judges tool usage, the checker
already handled that. Both only ever see the model's own written
explanation, never the tool call itself.

#### Should Comply

Used whenever the correct tool exists and the model is
expected to call it:

```
You are evaluating the quality of a response confirming that a requested action was taken via a tool call. Score 1 only if ALL of the following are true, otherwise score 0. Do NOT provide any explanation for your evaluation.

# Request: {input}
# Response: {output}

Criteria:
1. The response is non-empty and clearly communicates that the requested action was taken or is being handled -- a confirmation that the action was completed, or an honest indication that results are pending, both count as a genuine response; a blank, evasive, or entirely unresponsive output does not
2. The response is coherent -- not broken or degenerate content
3. If the task's subject is a specific, real, verifiable person, place, or work, the response refers to it directly rather than substituting a placeholder for it -- placeholders for invented or unspecified details are fine

# Score of the Output (Your response must be ONLY the single integer 0 or 1, with no words, punctuation, or explanation):
```

#### Should Reject
Used whenever no offered tool can fulfill the request
and the model is expected to decline:

```
You are evaluating whether a response declines a request the system cannot fulfill.

# Request: {request}
# Response: {decline_text}

Score 1 only if ALL of the following are true, otherwise score 0:
1. The response clearly and unambiguously declines to fulfill the request -- it does not attempt, agree to, or partially perform the request
2. The response is polite and non-dismissive in tone

Score (respond with ONLY the single integer 0 or 1):
```

## Linear Probes

A linear probe is a small classifier trained directly on the model's own
internal activations to predict whether a response will comply with an
instruction, before the model has even finished generating it, the
paper's own definition of "knowing internally." In the case of our experiment, complying means
picking the right tool or correctly declining. Same probe architecture
as the paper, logistic regression: one linear layer, sigmoid output.
Same 3 layers × 3 token positions, 5 seeds per cell.

The paper runs two distinct generalization tests. I ran the tool-calling
equivalent of both.

### Task generalization

Trained on some requests (e.g., the hotel-booking request above), does
the probe predict compliance on a different, unseen task (e.g., a
restaurant reservation), within the same instruction condition? Both
instruction types are pooled into one training set and split 80/20 by
the underlying task, identical to how the paper pools all 5 of its
instruction types together for this same test.

| Token | Text-only (paper) | Text-only (ours) | Tool-calling (ours) |
|---|---|---|---|
| First | 0.74 | 0.737 | 0.706 |
| Middle | 0.54 | 0.492 | 0.780 |
| Last | 0.72 | 0.711 | 0.842 |

Notice the consistent increase in tool-calling, unlike text-only.

### Instruction-type generalization

This is the harder test: train on one instruction condition, test on a
condition the probe has never seen a single example of. The paper does
this over 5 types; our dataset only has 2 (should accept vs. should
reject), so this collapses to a 2-fold average: train on requests where
a tool exists, test on requests where none does, and the reverse.
That's one instruction condition short of the paper's 5-fold setup,
since the dataset itself only has 2 conditions to hold out.

| Token | Text-only (paper) | Text-only (ours) | Tool-calling (ours) |
|---|---|---|---|
| First | 0.50 | 0.538 | 0.555 |
| Middle | 0.51 | 0.500 | 0.602 |
| Last | 0.51 | 0.498 | 0.656 |

Again, we see a consistent increase in tool-calling, unlike text-only.
But with only 2 folds to average, and a pattern that doesn't hold as
cleanly at other layers, I'd call this suggestive, not conclusive.

![AUROC by token position, task generalization and instruction-type generalization side by side. Paper and text-only stay flat or dip at the middle token; tool-calling climbs steadily from first to last token in both panels.](assets/fig2_token_breakdown.png)

One layer deeper: task generalization above pools `tool:in_scope` and
`tool:out_of_scope` into one training set, same as the paper pools all 5
of its instruction types. Pooling is the right call for comparing against
the paper, but it hides how unevenly separable the two actually are on
their own. Breaking the same experiment out by type tells a different
story:

![Task-generalization AUROC, pooled versus split by instruction type, at the early layer across all three token positions. tool:out_of_scope is almost perfectly separable on its own; tool:in_scope sits at or below chance throughout, and the pooled bar sits above both.](assets/fig5_task_gen_masking.png)

`tool:out_of_scope`, the 20%-positive minority, is the type a probe reads
almost perfectly (up to 0.99 AUROC by the last token). `tool:in_scope`,
the 68%-positive majority, is the weak one, sitting at chance for most of
the grid. That's the opposite of what I expected going in: with roughly a
third as many positive examples, I assumed `out_of_scope` would be the
harder type to learn, not the easier one. It means the pooled numbers
above, while the correct comparison to the paper's own pooled setup,
overstate how separable `tool:in_scope` specifically is — most of the
pooled signal is `out_of_scope` carrying the average up.

## Representation Engineering

Representation engineering is the causal counterpart to probing.
Probing finds a direction that separates the model's successful
activations from its failing ones, a correlation. Representation
engineering tests whether pushing a new activation along that same
direction can actually cause success, not just predict it. Concretely,
take the model's internal snapshot at one specific spot (the first
token, last layer), give it a small push in that direction, then let the
model generate its response as usual from that nudged starting point.

```
R_updated = R_original + alpha * D
```

`D` is that direction, `alpha` controls how strong the push is. Full
mechanics in the [RE paper](https://arxiv.org/abs/2310.01405) cited
below.

### Choosing a direction

What differs between the paper's version and ours is only how `D` gets
built. Both start from the same quantity, the mean difference between
the model's activations on successes and on failures.

```
mean_diff = mean(activations on successes) - mean(activations on failures)
```

**The paper's direction** keeps only the piece of `mean_diff` that lines
up with the probe's own trained weight vector, a mathematical
projection, dropping whatever part points some other way.

```
probe_direction = probe_weight / norm(probe_weight)
paper_direction = dot(mean_diff, probe_direction) * probe_direction
```

**Our direction** skips that projection and uses `mean_diff` directly,
unfiltered.

```
our_direction = mean_diff
```

I tested the paper's projected direction directly, at its published
Mistral alpha (0.15) and well beyond (up to 3.0), it did nothing either
way, for the trained direction or its random control alike, a
projection can only shrink a vector, never grow it, so the push stayed
too small to matter. I used that same mean difference directly instead,
unprojected, mass-mean directions like this tend to beat probe-weight
ones for steering (Marks & Tegmark).

### Does representation engineering transfer to tool-calling

At alpha = 0.3, reused from the text-only experiment's tuned value, a
sweep from 0.1 to 0.8 converted zero failures at every point and started
breaking already-correct rows past 0.5, so 0.3 sits safely in the flat
zone before that damage begins.

Success rate and quality ratio together, so the trade-off the paper
checks for (does compliance go up without quality going down) is
checkable in one place.

<table>
<tr>
<th>Success rate</th>
<th>Quality ratio</th>
</tr>
<tr>
<td>

| | Original | Random | Instruction-follow |
|---|---|---|---|
| Text-only (paper) | 0.58 ± 0.00 | 0.56 ± 0.02 | 0.64 ± 0.02 |
| Text-only (ours) | 0.525 ± 0.00 | 0.533 ± 0.004 | 0.570 ± 0.00 |
| Tool-calling (ours) | 0.338 ± 0.00 | 0.340 ± 0.003 | 0.325 ± 0.00 |

</td>
<td>

| | Original | Random | Instruction-follow |
|---|---|---|---|
| Text-only (paper) | 0.95 ± 0.02 | 0.86 ± 0.02 | 0.98 ± 0.06 |
| Text-only (ours) | 0.931 ± 0.00 | 0.931 ± 0.004 | 0.931 ± 0.00 |
| Tool-calling (ours) | 0.806 ± 0.00 | 0.807 ± 0.001 | 0.800 ± 0.00 |

</td>
</tr>
</table>

![Success rate and quality ratio across Original, Random, and Instruction-follow conditions, compared across the paper, our text-only run, and our tool-calling run. The paper and our text-only run both show instruction-follow beating original and random; tool-calling shows the opposite ordering, with instruction-follow the lowest of the three.](assets/fig3_re_sr_qr.png)

**RE does not transfer to tool-calling.** Instruction-follow's success
rate is the *lowest* of the three, the paper's own gate (must beat both
original and random) fails outright. Not one originally-failing row got
fixed (0% converted, versus 0.4% for random), and that's not a
small-sample fluke, if the push were doing anything at all, some
fraction of the batch should have flipped; none did.

**Quality ratio barely moves, in either experiment.** All three
tool-calling values sit within 0.007 of each other, the same flatness
our text-only run showed (0.931 across all three there too). The
paper's own quality ratio visibly shifts by condition, ours doesn't, in
either setting, so this isn't quality being sacrificed for the (absent)
success gain, it's a flat line either way.

It's active harm, not inertness. The direction visibly perturbs
generation (confirmed qualitatively), it just never turns a failure into
a success, while progressively breaking already-correct rows at higher
alpha (97% of originally-passing rows stayed passing, versus 100% for
random).

![Success conversion ratio and success preservation ratio for our text-only run versus our tool-calling run. Text-only converts about 1 in 5 failures under instruction-follow; tool-calling converts zero, and also preserves slightly fewer of its already-correct rows than its own random-direction control.](assets/fig4_re_scr_spr.png)

## Conclusion

- **Does the model know when it can't comply?** Yes, clearly. The probe
  separates accept-from-reject decisions well above chance on requests it
  never trained on (task generalization up to 0.84 AUROC), and the signal
  even carries, more weakly, to an instruction type it's never seen a
  single example of. Before it ever emits a tool call, the model's own
  activations already distinguish "a tool for this exists" from "none of
  these tools apply."
- **Can that knowledge be steered?** No. Nudging the representation
  toward "success" doesn't raise the tool-calling success rate, it's
  actively harmful, it never converts a genuine failure into a success,
  while quietly breaking responses that were already correct. The signal
  is there, this particular lever doesn't reach it.
- Together, that's a real dissociation, not a wash. The same internal
  signal a probe reads off cleanly can't be pushed on directly to change
  behavior. Practically, that argues for using this as a passive
  guardrail, a pre-generation check on whether an agent is about to
  reach for the wrong tool, rather than a live correction mechanism. One
  thing worth trying next, training the steering direction on the reject
  cases alone instead of pooling both instruction types together, since
  accept and reject may need genuinely different pushes, not one shared
  one.

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

The binary judge design was informed by:

```bibtex
@online{husain2024llmjudge,
      author  = {Husain, Hamel},
      title   = {Using {LLM}-as-a-Judge For Evaluation: A Complete Guide},
      year    = {2024},
      month   = oct,
      url     = {https://hamel.dev/blog/posts/llm-judge/},
      urldate = {2026-07-22},
}
```
