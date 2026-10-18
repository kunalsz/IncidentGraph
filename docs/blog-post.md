# I built an incident investigator that has to show its work

It is 2am. A dashboard lights up: checkout latency is climbing. Someone posts in the channel, "checkout seems slow, anyone know why?" and a dozen people start opening a dozen tabs. Metrics, logs, traces, the deploy log, the runbooks, the on call rota. Twenty minutes later someone finds the answer, and it was almost always one of two things: a change nobody remembered, or a dependency two hops away that nobody looked at.

That gap between "we have all the data" and "we know what happened" is the problem I wanted to poke at for the FalkorDB hackathon. I called the project IncidentGraph, and it is an AI investigator that runs on a graph instead of on a pile of files.

## The problem

The pitch for LLM driven root cause analysis sounds great until you try it. Ask a model "why is checkout slow" and give it your telemetry, and it will produce a confident paragraph. The problem is that the paragraph is often wrong, and worse, it is *plausibly* wrong. It will invent a metric value, or blame a service that is completely healthy, or connect two things that are correlated only because they both spiked at the same minute. In an incident, a confident wrong answer is more expensive than no answer, because someone acts on it.

So the thing I actually wanted to build was not "an LLM that investigates incidents." It was "an investigation that can prove what it claims, and that refuses to claim what it cannot prove." Those are very different products.

There is a second, quieter problem. Incident data is relational by nature. Services call services. Alerts observe services. Deploys target services. Runbooks apply to services. Past incidents affected services. A flat pile of logs has no opinion about any of that, which is exactly why root cause analysis is hard. A graph does have an opinion. That is what drew me to FalkorDB for this.

I entered this in Track 01, Agents That Act on Connected Data, because in this project the graph is not decoration. It is the thing that decides what the agent is allowed to see, the thing the deterministic scores are computed inside, and the thing every claim gets checked against before it reaches you.

## What IncidentGraph does

You give it a symptom ("checkout latency up") or an alert id. It builds a small, private graph for that one investigation, scopes it down to the anchor service and its immediate dependencies plus the relevant telemetry, scores every suspect without calling a model at all, and only then lets an agent reason inside that bounded picture. At the end it produces a leading root cause hypothesis, an evidence path you can trace link by link, and a recommended next action, with a numeric citation on every claim.

The engine runs as seven phases, and the shape of them is the whole point:

- **P0 resolves** the symptom to one anchor service. If the name is ambiguous, it stops and asks instead of guessing.
- **P1 scopes**. This is a `GRAPH.COPY` of the session graph, then a Cypher projection of the anchor plus one hop of dependencies, capped at 40 nodes and 12 metric series. It computes severity, span self time, three sigma flags, and deploy proximity. Zero LLM calls.
- **P2 ranks** the suspects deterministically. Instability, self processing impact, severity, a random walk rooted at the alert, and change proximity, combined with reciprocal rank fusion. Zero LLM calls.
- **P3 investigates** the shortlist with bounded agents, one bundle per candidate, hypotheses persisted into the graph as they are made.
- **P4 validates**. Every citation is re resolved against the graph from scratch, and claims that violate a deterministic check are vetoed.
- **P5 fuses** everything into one ranked list with typed confidence, and it never lets a weighted score override a veto.
- **P6 reports**, and decides the confidence tier in code.

The through line across all seven is one rule I kept repeating to myself: the model proposes, the graph and the code dispose.

## Five things I learned

**1. A graph is a boundary, not just a database.** The single best decision of the whole project was to make the graph the thing that bounds the agent instead of the thing the agent queries freely. Every phase reads an immutable snapshot of a small projected subgraph. The agent cannot wander off into the full topology or invent a service that is not in scope, because scope is a number, not a suggestion. Before I did this, the agent's token usage was unpredictable and its answers were unfalsifiable. After, it was auditable. If you are building an agent on connected data, I would start here.

**2. Do the deterministic work first, then let the model adjudicate.** Every score that can be computed in Cypher gets computed before an LLM sees anything. Severity, self time, the sigma flags, the random walk mass, the fusion. By the time a model shows up, its job is to explain a shortlist, not to rank from scratch. This cut cost, but the real win was reproducibility. When the ranking is deterministic, a wrong answer is a bug I can find. When it is a model's opinion, a wrong answer is a mystery.

**3. A prompt is not an invariant.** I could write "only say CONFIRMED if every causal link has supporting evidence" in a prompt forever and it would sometimes be ignored. So the confidence gate is code, not prose. `CONFIRMED` is refused unless every dependency association in the evidence path carries at least one supporting evidence node *and* every deterministic validator passed. The model writes the summary. A separate module decides the tier, and that module does not care how confident the model sounds. The first time the gate downgraded a report I was annoyed. Then I realised that is the feature working.

**4. Fail closed, and treat abstention as a real answer.** If the anchor is ambiguous, stop. If the winner cannot be re verified against the live graph at report time, drop it and say why. If a modality has records but none of them carry a usable score, do not report a zero severity and let everyone read it as "normal." Write an explicit gap instead. A lot of the code I am proudest of is the code that says "I do not know." In incident response that is more useful than a guess, because a human can act on a known gap and cannot act on a false certainty.

**5. The measurement surprised me, and that is the point.** I built a fairly elaborate five component ranking, and when I ran it against the RE2-OB benchmark from RCAEval, a much simpler policy (rank on severity alone) beat it badly on a held out split. I have not switched the default, because it was a case level split and one experiment is not a mandate, but it was a good reminder that a transparent baseline earns its keep. If you are going to publish a fusion formula, you had better be willing to be embarrassed by the ablation.

## The honest part

I spent the majority of my time on the machinery that makes the output trustworthy, which means I shipped less of the flashy agentic stuff than I planned. Reflection and escalation between branches are built but currently switched off in the version I ran. The multi hop tool calling investigator is a fixed two prompt pipeline right now, not the tool using agent I sketched on day one. Cross incident memory recall reads from the graph but does not yet write back the links that would make it grow.

I would defend that tradeoff, because an investigator that cannot be trusted is not useful at 2am, but it is a tradeoff and not a triumph. If I keep going, the next thing I build is not a smarter model. It is more of the same discipline applied to the parts I skipped.

## What I am taking away

Connected data changed what an agent can be honest about. On a flat store, "why" is a story the model tells. On a graph, "why" is a path, and a path can be checked edge by edge, and a claim that has no path behind it can be refused. That is the whole idea, and it is why FalkorDB was not just the storage choice for this project but the reason the project makes sense at all.

If you want to read the code or run it locally, the repository has the full setup instructions and the demo fixtures. It runs against a local FalkorDB, seeds a small astronomy shop topology, and you can watch a single incident move through all seven phases and come out the other side with an evidence path or an honest refusal.

Thanks to FalkorDB and WeMakeDevs for running the hackathon. Most of what I learned here I learned by watching a gate refuse a report I wanted it to accept.
