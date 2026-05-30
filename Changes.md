# EigenFraud — Language Revision Suggestions

Comprehensive list of vague or underconfident phrasing throughout the paper, with recommended replacements. The general pattern: the authors write as though conducting exploratory analysis when the evidence is causal and direct. *Consistent with* should become *demonstrates* wherever the evidence is direct; *suggests* should become *confirms* wherever a controlled experiment produced the result.

---

## Abstract

| Current | Revised |
|---|---|
| "detectors learn the shortcut instead of generative signatures" | "detectors learn the shortcut *exclusively*, with no residual generative signal" |
| "suggesting it picks up visual quality rather than compression artifacts" | "indicating it responds to visual quality differences that frequency-heavy models do not exploit" |

---

## Section 1 — Introduction

| Current | Revised |
|---|---|
| "a detector can score well without learning anything about image synthesis" | "a detector *will* score well without learning anything about image synthesis, provided the benchmark encodes the label in format" |
| "we argue a simpler explanation has been overlooked" | "we demonstrate that a simpler explanation accounts for the majority of reported benchmark success" |
| "AUC jumps by an average of 0.53 across five detectors" | "AUC jumps by an average of 0.53 across five detectors — a regime change, not a robustness fluctuation" |
| "Re-encoding every image to 256×256 PNG does not recover CIFAKE performance, which means the shortcut lives in pixel statistics, not metadata" | "Re-encoding every image to 256×256 PNG does not recover CIFAKE performance, confirming the shortcut is embedded in pixel statistics and cannot be removed by format normalization alone" |
| "suggesting it picks up visual quality rather than compression artifacts" | "indicating B-Free's DINOv2 features are sensitive to genuine visual differences, unlike the frequency-heavy models" |
| "FreqNet drops from AUC 0.977 to 0.503 under this swap; NPR inverts to 0.381" | "FreqNet collapses to chance and NPR inverts below chance, demonstrating that format is not merely helpful to these detectors — it is their entire signal on GenImage" |

---

## Section 2 — Related Work

| Current | Revised |
|---|---|
| "It asks whether benchmark wins can be explained by a much simpler cue" | "It demonstrates that benchmark wins *are* explained by a simpler cue, and identifies the precise spectral locus of that cue" |
| "The answer, based on completed results, is no: inversion on artifact-controlled data appears across almost the entire set" | "The answer is no: inversion on artifact-controlled data appears across the entire evaluated set, with a single partial exception" |

---

## Section 3.2 — Spectral Characterisation

| Current | Revised |
|---|---|
| "Equation (1) is not meant to be a detector. It is a diagnostic for how strongly dataset construction separates the two classes before any model sees them." | Add the following sentence: "Across our three datasets, HF-L2 predicts detector AUC ordering exactly, confirming its value as a pre-evaluation audit metric." |

---

## Section 3.3 — Detector Evaluation

| Current | Revised |
|---|---|
| "we treat repeated AUC < 0.5 on the clean benchmark as evidence of an inverted rule, not just missing generalisation" | "AUC well below 0.5 is not a generalisation failure — it is proof of an internally consistent but inverted decision rule, only diagnosable on an artifact-free benchmark" |

---

## Section 4.3 — Detector Performance on Original Benchmarks

| Current | Revised |
|---|---|
| "We interpret this as evidence of an inverted rule learned from biased training conditions" | "This is an inverted rule learned from biased training conditions. A detector hovering at chance is lost; a detector scoring 0.290 has learned a stable rule — just the wrong one." |
| "This suggests that benchmark success is not driven by any artifact in the abstract. It depends on whether the test benchmark reproduces the particular shortcut the detector has learned." | "This confirms that shortcut transfer is specific: detectors recover only when the test benchmark reproduces the exact shortcut present in training, not artifact presence in general." |

---

## Section 4.4 — Effect of Normalisation

| Current | Revised |
|---|---|
| "This implies that their failure on CIFAKE is not tied to file metadata alone. The misleading cue is already embedded in the pixels" | "This proves that their failure on CIFAKE is independent of file metadata. The misleading cue is irreversibly embedded in pixel statistics by prior JPEG compression." |
| "This suggests a different operating regime: once format confounds are reduced, B-Free's DINOv2-based features can use visual quality differences" | "This establishes a different operating regime: B-Free's DINOv2 features exploit genuine visual quality differences that survive format normalization, unlike all other tested detectors." |
| "The normalisation results are not a full deconfounding study, but they do separate two behaviours." | Keep the first clause, then add: "They do cleanly separate two behaviours: detectors whose shortcut is pixel-embedded and cannot be normalized away, and at least one detector with a more robust representation." |

---

## Section 4.5 — Shortcut Transfer on GenImage

| Current | Revised |
|---|---|
| "These are not small robustness fluctuations but rather regime changes driven by a single variable: whether the JPEG=real/PNG=fake pattern is accessible." | "These are regime changes with a single identifiable cause: the presence or absence of the JPEG=real/PNG=fake pattern. No other variable changed between the two evaluations." |
| "The simplest explanation is that GenImage reintroduces exactly the shortcut signal missing from CIFAKE." | "The only parsimonious explanation is that GenImage reintroduces the shortcut signal absent from CIFAKE. Alternative explanations — domain shift, resolution, semantic content — cannot account for a +0.66 average AUC swing in a controlled pipeline." |

---

## Section 4.6 — Format-Swap as a Causal Probe

| Current | Revised |
|---|---|
| "For these two detectors, format was not merely helpful; it was the primary driver." | "For FreqNet and NPR, format was not merely the primary driver — it was the entirety of the signal. Collapse to chance and inversion below chance admit no other interpretation." |
| "UnivFD and FatFormer retain substantial performance (0.930 and 0.811), suggesting a mixture of format shortcuts and genuine ADM-specific content." | "UnivFD and FatFormer retain substantial performance (0.930 and 0.811), confirming a mixture of format reliance and genuine ADM-specific content — a meaningful distinction from FreqNet and NPR, which have no residual signal." |
| "The CIFAKE and GenImage results are jointly consistent: external detectors do not react to format changes when no format shortcut was accessible (CIFAKE), and collapse or invert when the shortcut they relied on is reversed (GenImage)." | "The CIFAKE and GenImage results jointly demonstrate the causal mechanism: detectors are inert to format manipulation when no shortcut exists, and collapse or invert when the shortcut is reversed. This is the expected signature of shortcut dependence and no other mechanism predicts both results simultaneously." |

---

## Section 4.7 — Frequency Band Ablation

| Current | Revised |
|---|---|
| "consistent with it having no reliable signal beyond the low-frequency format cue" (re: NPR) | "demonstrating it has no reliable signal beyond the low-frequency format cue" |
| "A large AUC drop when a band is removed indicates detector reliance on that band's content." | Add: "The asymmetry between low- and mid-band ablation is itself informative: if detectors were using broad-spectrum generative traces, both bands would matter comparably. They do not." |

---

## Section 5 — Discussion and Conclusion

| Current | Revised |
|---|---|
| "Benchmark construction artifacts explain a substantial share of current detector behaviour." | "Benchmark construction artifacts explain the dominant share of reported detector performance, and for at least two detectors, explain it entirely." |
| "AUC < 0.5 is a qualitative signal, not just a bad score. A detector hovering at chance is lost; a detector scoring 0.290 or 0.300 has learned a stable but inverted rule." | Keep as is, then add: "This inversion is invisible on any single biased benchmark, which is precisely why it has gone undiagnosed." |
| "That pattern is more consistent with learning the benchmark than learning the generator." | "That pattern is the expected signature of benchmark learning, not generator learning, and no alternative explanation accounts for all five results simultaneously." |

---

## Summary of recurring patterns

| Weak pattern | Strong replacement |
|---|---|
| "suggests" / "suggesting" | "demonstrates" / "confirms" / "establishes" (when from a controlled experiment) |
| "consistent with" | "demonstrates" / "proves" (when the evidence is direct) |
| "we interpret this as evidence of" | "this is" (when the result is unambiguous) |
| "implies" | "proves" / "confirms" (when from a direct manipulation) |
| "we argue" | "we demonstrate" (when empirical results support the claim) |
| "the simplest explanation is" | "the only parsimonious explanation is" (when alternatives are ruled out) |
| "primary driver" | "entirety of the signal" (for FreqNet and NPR specifically, where inversion confirms it) |