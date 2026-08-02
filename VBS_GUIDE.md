# The Video Browser Showdown (VBS) — A Complete Guide

---

## Table of Contents

1. [What VBS Is](#1-what-vbs-is)
3. [Competition Format](#3-competition-format)
4. [Task Types](#4-task-types)
5. [Scoring and Evaluation](#5-scoring-and-evaluation)
6. [The Evaluation Infrastructure: DRES](#6-the-evaluation-infrastructure-dres)
7. [Interaction Logging](#7-interaction-logging)
8. [Datasets](#8-datasets)
9. [Provided Analysis Data and Starter Tools](#9-provided-analysis-data-and-starter-tools)
10. [Participating Systems and Technical Trends](#10-participating-systems-and-technical-trends)
11. [How to Participate](#11-how-to-participate)
12. [Relationship to Other Benchmarks](#12-relationship-to-other-benchmarks)
13. [Where to Find Things](#13-where-to-find-things-quick-links)
14. [References](#14-references)

---

## 1. What VBS Is

The Video Browser Showdown is an international competition that evaluates the state of the art in **interactive video retrieval**. It has run annually since 2012 as a special session/side event at the **International Conference on MultiMedia Modeling (MMM)** [1][2].

### The core idea

Most retrieval benchmarks evaluate an algorithm in isolation: you submit a ranked list, and a metric scores it. VBS does something different — it evaluates **a human plus a system, working together, under time pressure, in the same room**. This matters because real video search is rarely a one-shot query. A journalist hunting a half-remembered clip in an unannotated archive will query, look, reformulate, browse, and query again. VBS is designed to measure how well a tool supports that whole loop [3].

### What makes it hard

- **The dataset is known; the queries are not.** Teams receive the video collection months in advance and can index and analyze it however they like. The actual search tasks are generated and issued *live, on-site*, and are unknown until the moment they appear on the projector [1].
- **Time limits are tight.** Typically 5 minutes per task, 7 for textual known-item search [4].
- **Wrong answers hurt.** False submissions carry an explicit scoring penalty, so teams cannot brute-force by submitting everything plausible [1][4].
- **The collections are large.** The current V3C collection spans roughly 3,800 hours across 28,450 videos and over 4.1 million shots [5].

### Why it matters to the field

Because winning approaches tend to set the research agenda for the following years, VBS functions as more than a leaderboard — it steers the direction of interactive video retrieval research. The competition documented the field's shift from handcrafted similarity models, to deep concept detectors, to the current dominance of joint text-visual embedding models such as CLIP [4].

## 3. Competition Format

### The session

VBS is a **moderated live event**. Tasks are projected on a screen at the front of the room. Teams sit at their own machines running their own systems, and submit answers over the network to the evaluation server. A live scoreboard on the same projector shows every team's running score and submission history in real time — correct submissions in green, wrong ones marked as such [9][13].

Historically, the format was:

- **Expert session** — the system developers themselves operate their tools on the full range of tasks.
- **Novice session** — volunteers recruited from the conference audience operate the teams' systems on typically easier tasks, testing usability by non-experts [6].

The novice session has since been removed [13], though its purpose — measuring usability rather than just capability — is still recognized as valuable and its removal is noted as a loss in the community's own analysis [4].

### Team composition

Teams have typically fielded up to two operators per system [4]. How multiple users are aggregated has changed:

| Model | How it works | Trade-off |
|-------|--------------|-----------|
| Collaborative single team (≤2023) | Task solved when the fastest team member submits correctly; others' wrong submissions still penalize | Penalizes single-operator teams; hides individual performance; searching stops once one user succeeds, losing data |
| Independent users, averaged | Each user solves independently, scores averaged | Requires equal team sizes, which cannot be guaranteed |
| **Independent users as distinct teams (2024→)** | Each user is scored as their own team; system's rank is that of its best user | Any number of users can join without unfairness; enables variance analysis and "super-user" identification |

The third option was adopted from VBS 2024 onward [4].

### Rules

- No restrictions on features or techniques — fully automatic search is permitted alongside interactive search [13].
- **Screen recording of the task presentation is not allowed** (this would let a system re-query the target clip directly) [13].
- Teams must implement REST API submission and logging against the evaluation server [13].

---

## 4. Task Types

There are three families of task, with several variants. The mix is deliberately weighted toward the harder types [13].

### 4.1 Known-Item Search (KIS)

There is exactly **one** correct segment in the entire collection. Find it.

| Variant | How the target is presented | Notes |
|---------|---------------------------|-------|
| **KIS-V** (visual) | A short clip (a few seconds) selected at random from the dataset is played on the projector | The "easy" variant — you can see exactly what you're looking for. Now deliberately a minority of tasks [13] |
| **KIS-T** (textual) | Only a text description read out and displayed by the moderator; no visuals | Much harder — exposes the semantic gap between language and imagery. Usually gets 7 minutes rather than 5, and the description is often extended during the working time [4] |
| **KIS-C** (chat / conversational) | Starts as a minimal textual description; after 60 seconds, further details are revealed in response to questions and chat from participants | Models a realistic scenario where a searcher progressively elicits detail from a person who remembers a clip vaguely |

KIS-V variants are also run on non-V3C collections, e.g. **KIS-V-M** for marine video tasks [4].

Why textual KIS is genuinely difficult: in the VBS 2023 analysis, there were tasks where the correct item appeared in a team's top-10 results but users still couldn't find and submit it within 100 seconds, and at least one case where a team never submitted despite having the correct video ranked first [4].

### 4.2 Ad-hoc Video Search (AVS)

A broad textual description matching an **undetermined number** of shots — e.g. *"find all shots showing cars in front of trees"* or *"find shots showing a bar chart"* [1][4]. Teams find and submit as many distinct matching segments as possible.

The critical scoring twist: **diversity matters more than volume**. Many temporally-close shots from the same video count for far less than shots drawn from different videos [1]. Under current scoring, submitting a second correct shot from a video you've already scored adds nothing at all [4].

Because there is no pre-existing ground truth for such open-ended queries, AVS submissions are **judged live during the competition** by experienced human judges via the evaluation server [4].

### 4.3 Visual Question Answering (VQA / QA)

A question is asked about a specific video or collection, and the answer is submitted as manually entered text — e.g. *"How many nights do we see passing in the video until this segment?"* [1]. This cannot be solved by retrieval alone; it requires actually inspecting and reasoning about the content. It is the newest task family and reflects the competition's expansion beyond pure retrieval [4].

---

## 5. Scoring and Evaluation

Scoring rewards **speed** and **accuracy**, and punishes **guessing**. The two task families use different formulas.

### 5.1 KIS scoring

For a task with a 300-second limit, a correct submission earns 0–100 points made up of [14]:

```
score = 50                    (base reward for solving the task at all)
      + (300 − t) / 6         (time bonus; t = elapsed seconds at submission)
      − 10 × |WS|             (penalty; |WS| = number of wrong submissions
                               before the first correct one)
```

Reading the formula: solving the task instantly earns the full 100; solving it at the buzzer earns 50; and **each wrong guess costs 10 points — equivalent to burning 60 seconds** of the clock. That ratio is what makes reckless submission a losing strategy.

Under the collaborative team model, the team's time was the time of the *first correct* submission by any member, but *all* wrong submissions by any member before that point were penalized [14].

### 5.2 AVS scoring

VBS 2023 introduced a diversity-aware formula. A team `t` scores [4]:

```
f_t = 1000 · max( (1/|C|) · Σ_{v ∈ V_t} (c_v − i_v · p) , 0 )

where:
  C   = the set of correct videos found across ALL teams' submissions
  V_t = the set of videos team t submitted for
  c_v = 1 if team t has a correct submission for video v, else 0
  i_v = number of incorrect submissions for v before the first correct one
  p   = penalty constant, set to 0.2
```

Two design consequences worth understanding:

- **Per-video credit, not per-shot.** Multiple correct shots from one video earn credit once. This directly implements the "diversity beats quantity" principle.
- **The denominator is a shared pool.** Because each team is divided by `|C|` — the union of correct videos found by *everyone* — finding a video that no other team found is disproportionately valuable, while everyone's score is diluted by videos the field collectively discovers.

When this formula was compared retroactively against the old one, the resulting team rankings correlated at 0.929 — i.e. it did not substantially change who won, and teams' strategies appeared largely unaffected by it [4].

### 5.3 Normalization across categories

Overall standings are computed per task category, with the best-performing team in each category normalized to 1,000 points and all others scaled proportionally. A team's overall score is the sum across categories [4]. This prevents any single task type from dominating and rewards all-round systems.

### 5.4 Live judging

For AVS (and any task without pre-defined ground truth), human judges assess submissions in real time through the server's judging interface. Because teams submit in different units — a frame number, a timestamp, a predefined shot ID — the server maps every submission onto **predefined reference shots** before presenting them to judges, so judgments are consistent [4].

Judging is imperfect and the community measures this openly: the median verdict takes about 10 seconds, and teams sometimes disagree with judges. In one 2023 task, seven different teams submitted the same shot and all were judged wrong — the shot showed one person riding a horse in its first half and several riders in the second, and teams had likely judged from a keyframe while the judge assessed the whole shot [4]. Since 2022, AVS queries are reviewed with the judging team and dry-run in advance to reduce ambiguity [4].

---

## 6. The Evaluation Infrastructure: DRES

**DRES — the Distributed Retrieval Evaluation Server** — is the open-source system that runs the competition [15]. It builds on the earlier VBS server and is developed primarily at the University of Basel and University of Zurich [11][15].

### What it does

DRES has two jobs [1]:

1. **Presents all tasks** on the projector (and in-browser for remote teams), including the countdown and the live scoreboard.
2. **Collects and evaluates all submissions** from every participating system — checking against ground truth where it exists, and routing to live judges where it does not.

### Why it exists

Before DRES, interactive retrieval evaluation required every system to be physically present in the same room at the same time. That constrains organization, caps how many systems can be evaluated in a fixed time slot, and collapses entirely under travel restrictions. DRES was built to support both traditional on-site and fully distributed evaluation [11]. A later extension relaxed the *simultaneity* constraint as well, allowing participation from any place at any time within a defined window — closing much of the gap between interactive and non-interactive campaigns [16].

DRES is now used well beyond VBS: also by the **Lifelog Search Challenge (LSC)** and by challenges such as the Ho Chi Minh City AI Challenge [17].

### How systems talk to it

Communication is over a simple **HTTP protocol**, formally described by an **OpenAPI specification** [1][13]. The recommended workflow is to generate client classes automatically from that spec rather than hand-writing HTTP calls. Generated clients cover **login, logout, result submission, and logging** [1].

The DRES developers maintain a **Client-Examples** repository with working examples in several programming languages [1][13].

**Requirements:** DRES needs a JRE of at least Java 11 to deploy; development additionally uses NPM and the Angular CLI [15].

### Worked example: generating an Angular client

*(from the source document; useful as a concrete template)*

Install the generator tooling:

```bash
npm install @openapitools/openapi-generator-cli -g
npm install -g ng-openapi-gen
```

Add a generation script to `package.json`:

```json
"scripts": {
  "gen-dres-client": "openapi-generator-cli generate -g typescript-angular -i https://raw.githubusercontent.com/dres-dev/DRES/2.0.1/doc/oas-client.json -o openapi/dres --skip-validate-spec --additional-properties npmName=@dres-client-openapi/api,ngVersion=13.0.0,enumPropertyNaming=original"
},
"dependencies": {
  "@openapitools/openapi-generator-cli": "2.4.26"
}
```

Generate the TypeScript client:

```bash
npm run-script gen-dres-client
```

Then import the generated services:

```typescript
import { SubmissionService } from '../../openapi/dres/api/submission.service';
```

> Pin the version in the spec URL to whichever DRES release the competition is running, rather than assuming `2.0.1`.

### Citing DRES

If you use DRES, cite the appropriate paper. The currently recommended citation is Sauter et al., *Performance Evaluation in Multimedia Retrieval*, ACM TOMM 2024 [15][17]; the original system paper is Rossetto et al., MMM 2021 [11].

---

## 7. Interaction Logging

This is easy to treat as an afterthought and shouldn't be — it is how the competition produces science rather than just a winner.

Every year the VBS community performs a **detailed post-hoc evaluation** of the competition, published as a joint journal paper. That analysis depends on teams submitting **interaction logs** to DRES alongside their answers [18]. The OpenAPI client provides methods for this; the log message format is defined in an actively maintained specification document [18][15].

**What gets logged:** query specifications, complete result lists (at least the top-*k* retrieved items per query), timestamps, team and user identifiers [4][15].

**Why it's worth the effort.** The VBS 2023 analysis used these logs to separate three things that a raw leaderboard conflates [4]:

- *Retrieval failure* — the system never surfaced the right item.
- *Browsing failure* — the item was in the top results, but the user didn't spot it in time.
- *Query formulation* — how users phrase, rephrase, and converge on queries.

Findings from that analysis are only possible with logs: longer text queries tended to rank targets better; query diversity was low within a single user but high between users of the same team, suggesting the *user*, not the tool, is the main source of query variation; and teams sometimes failed despite having the correct video ranked in their top ten, pointing straight at interface design rather than the model [4].

**Practical warning:** in 2023, several teams' logs were unusable — unrecoverable timestamps, incomplete records from logging failures — and those teams were dropped from the analysis. Teams also logged to inconsistent depths (10,000 results vs 5,000 vs 1,000) and in inconsistent units (frames vs. predefined shot IDs), which had to be normalized [4]. If you participate, treat logging as a first-class requirement, test it before the event, and log generously.

---

## 8. Datasets

### 8.1 V3C — Vimeo Creative Commons Collection

The primary VBS collection, provided by **NIST** in collaboration with **TRECVID** [13]. It ships with segmentation information — shot boundaries and keyframes — already computed [1].

| Shard | Videos | Duration | Predefined segments | Size |
|-------|--------|----------|--------------------|------|
| V3C1 | 7,475 | 1,000 h | 1,082,659 | 1.3 TB |
| V3C2 | 9,760 | 1,300 h | 1,425,454 | 1.6 TB |
| V3C3 | 11,215 | 1,500 h | 1,635,580 | — |
| **Total** | **28,450** | **~3,800 h** | **4,143,693** | **~8.7 TB** |

*Per-shard figures from [13]; totals as given in the source document and consistent with the sum.*

**Getting access:** complete the data agreement form at `http://www-itec.aau.at/~klschoef/VBS2019/V3C_Org.Form.txt`, and email a scan to `angela.ellis@nist.gov`, CC `gawad@nist.gov` and `ks@itec.aau.at`. NIST replies with the download link [1][13].

V3C is described in Rossetto et al., *V3C — A Research Video Collection* (MMM 2019) [19].

### 8.2 MVK — Marine Video Kit

Underwater and scuba-diving footage, contributed by Prof. Sai-Kit Yeung's group [13]. Introduced at VBS 2023 as a small but deliberately punishing collection: roughly **1,374 videos totaling about 12 hours** [4].

Small does not mean easy. MVK is characterized by highly redundant, noisy content shot with moving cameras in murky water — the visual distinctiveness that CLIP-style models rely on is largely absent [4]. It exposed real weaknesses: one team's coarse uniform frame sampling made many MVK queries simply unsolvable, and most teams showed one user clearly outperforming the other, suggesting effective search strategies for this domain are not yet established [4].

Since VBS 2025 there is a **second part (MVK 2.0)**. Both parts are available from `https://mvk.hkustvgd.com`, though the organizers recommend downloading the VBS-specific snapshot from the same FTP that hosts the medical data, since the competition uses a modified snapshot rather than the live dataset [1][4]. MVK is described in its MMM 2023 paper [20].

### 8.3 GynSurg (formerly LHE75 / VBSLHE) — Medical Video

Laparoscopic gynecology footage — **LHE** stands for *laparoscopic hysterectomies*. The original LHE75 release contained 75 videos totaling roughly 100 hours [1].

> ⚠️ **Naming update:** the dataset the source document calls **LHE75** is now referred to as **GynSurg** (also seen as GynSurgLHE or VBSLHE) [13]. Expect all three names in the literature.

Access is by SFTP. Contact **Klaus Schoeffmann** for the data agreement form; credentials follow after signing [1][13]. Described in its own 2025 paper [21].

### 8.4 Historical collections

For context when reading older VBS papers: 2015–2016 used the **BBC** collection from MediaEval Search & Hyperlinking; 2017–2018 used TRECVID's **IACC.3**; and the earliest editions used a small ad-hoc collection of about 30 long videos [2][6].

---

## 9. Provided Analysis Data and Starter Tools

The organizers deliberately lower the barrier to entry by publishing precomputed analysis and open-source baseline systems, so a new team doesn't need to build a full pipeline from scratch [1].

### Precomputed content analysis

V3C ships with shot boundaries and keyframes. On top of that, the organizers provide results from various content-analysis steps — color, faces, text, detected ImageNet classes, and more [1]:

- **V3C1 analysis data:** `https://github.com/klschoef/V3C1Analysis` — described in an accompanying article on records.mlab.no [1]
- **V3C2 analysis data:** `https://arxiv.org/abs/2105.01475` [1]
- **V3C1 ASR (speech recognition) data:** `https://github.com/lucaro/V3C1-ASR`, released by Luca Rossetto et al. [1]

### Shot boundary detection

**TransNet V2**, from the SIRET team (Jakub Lokoč's group), is a state-of-the-art shot boundary detection network — useful for datasets like MVK that don't ship with segmentation [1]:

- Code: `https://github.com/soCzech/TransNetV2`
- Paper: `https://arxiv.org/pdf/2008.04838.pdf`

### Baseline systems you can build on

| System | What it is | Where |
|--------|-----------|-------|
| **SOMHunter** | Lightweight open-source version of the **VBS 2020 winning system**, shipped with all necessary V3C1 metadata. The fastest route to a working competitive system | `https://github.com/siret/somhunter` [1][2] |
| **vitrivr** | Modular open-source multimedia retrieval stack, a long-running VBS participant and multiple-time winner (2019, 2021). Its flexible architecture makes it a research platform, not just an entry | `https://vitrivr.org/` [1][2] |

vitrivr comprises three components: the user interface (`vitrivr-ng`), the **Cineast** retrieval and feature-extraction engine, and the **Cottontail** database [4]. A VR variant, **vitrivr-VR**, also competes [4].

### Past tasks and results

The **VBS-Archive** repository holds tasks and results from previous competitions, plus AVS judgements in TRECVID format from 2021–2024, and links to every annual analysis paper [2]:

`https://github.com/lucaro/VBS-Archive`

---

## 10. Participating Systems and Technical Trends

### The CLIP consolidation

The single most important trend: by VBS 2023, nearly every system used a joint text-visual embedding model, most often some CLIP variant [4]. This produced a striking effect — **performance across teams became notably balanced**, because everyone had access to roughly the same retrieval power.

That, in turn, moved the competitive frontier elsewhere. Systems using the *same* CLIP model did not perform the same. What separated the top teams was [4]:

- combining **multiple** models and search modes rather than relying on one;
- the quality of the **browsing interface**;
- the ability to **reorder results** via temporal search and visual similarity.

Teams using newer OpenCLIP models trained on LAION datasets performed strongly, though teams on the original CLIP remained competitive [4].

### VBS 2023 final standings and capabilities

A useful snapshot of what a competitive system looks like [4]:

| Rank | System | Country | Score | KIS solved (of 19) |
|------|--------|---------|-------|--------------------|
| 1 | vibro | DE | 3,992 | 18 |
| 2 | VISIONE | IT | 3,625 | 17 |
| 3 | VIREO | SG | 3,258 | 16 |
| 4 | vitrivr-VR | CH | 3,200 | 16 |
| 5 | CVHunter | CZ | 3,027 | 13 |
| 6 | vitrivr | CH | 2,986 | 14 |
| 7 | Verge | GR | 2,803 | 13 |
| 8 | QIVISE | CN | 2,314 | 14 |
| 9 | VideoCLIP | IE | 1,858 | 9 |
| 10 | v-FIRST | VN | 1,773 | 9 |
| 11 | diveXplore | AT | 1,647 | 9 |
| 12 | 4MR | CH | 1,626 | 10 |
| 13 | Perfect Match | AT | 34 | 0 |

### The functional building blocks

Reading across systems, a competitive VBS entry generally combines several of these [4]:

**Search modes**
- *Joint text-visual embedding* — free-form text queries against visual content. Now the backbone of essentially every system.
- *Concept search* — object detection and scene classification (COCO, ImageNet, Places365), OCR, ASR. Note that in 2023 the top team used none of these text-based methods, and diveXplore reported concept search was overshadowed by CLIP's performance.
- *Query-by-example* — use a retrieved frame as the next query. Now typically implemented with the same CLIP visual encoder.
- *Temporal queries* — describe two scenes that occur in sequence, then fuse or window the two result lists. VISIONE used these heavily and it was one of the factors distinguishing it.
- *Relevance feedback* — iteratively refine from user-marked positives. Approaches ranged from Bayesian models (CVHunter) to quantum-inspired re-ranking (QIVISE) to AVS feedback loops (vibro).

**Browsing and interface**
- 2D similarity-arranged maps of images (vibro's signature feature)
- Results grouped by video, chronologically ordered within each
- Video players, previews, temporal context views, top-*k*-per-video filters
- VR interfaces — vitrivr-VR arranges results cylindrically around the user; despite often *not* having the target in its result set, it browsed its way to correct answers in nearly all KIS tasks and finished 5th

**Emerging directions**
- **LLM-assisted query suggestion** — v-FIRST pioneered this at VBS 2023, suggesting search terms to sharpen the user's query. It did not top the leaderboard, but the analysis flags the direction as promising [4].
- **Composed image retrieval** — combining a visual example with a textual modification, giving fine-grained control over the target [4].
- **Speech input** — VISIONE integrated Whisper so users could dictate queries [4].

### Open source

Among VBS 2023 systems, **VISIONE, vitrivr, and vitrivr-VR** were fully open source; others draw heavily on open models and repositories [4].

---

## 11. How to Participate

### Who can enter

Anyone with an exploratory video search tool supporting retrieval, interactive browsing, and exploration in a video collection [13].

### The requirements

1. **Submit an extended demo paper** — 6+2 pages in Springer LNCS format (the 2 extra pages are for references only). Submit through the MMM submission system, selecting the "Video Browser Showdown" track [13].
2. The paper must describe the tool in detail, including a screenshot, and explain how it supports interactive video search. Submissions are **peer-reviewed**, and accepted papers appear in the MMM conference proceedings [13].
3. **Implement DRES integration** — result submission and logging via REST API. The organizers note this "requires some efforts for integration," so budget time for it [13].
4. **Present your system** at the public session, usually via a short introductory video, sometimes a poster [13].

### Current restrictions (VBS 2027)

- No virtual attendance [13]
- No novice session — developers only [13]
- KIS-V tasks will be a minority; expect mostly KIS-T, KIS-C, VQA, and AVS [13]
- No screen recording of the task presentation [13]

### After the competition

A **joint journal paper** is written each year with contributions from every participating team. The winning team takes the lead as main author [13]. This is a genuine incentive beyond the trophy — it's a co-authorship on a well-cited annual survey of the field.

---

## 12. Relationship to Other Benchmarks

### TRECVID (NIST)

The relationship is collaborative rather than competitive. VBS uses the V3C collection **in collaboration with NIST**, aligned with TRECVID's own Ad-Hoc Video Search task [13]. The AVS task type is shared conceptually, and VBS publishes its AVS judgements in TRECVID format [2].

The distinction: TRECVID's AVS task evaluates automatic systems producing ranked lists; VBS evaluates a human interacting with a system live under a clock. They measure different things about the same underlying technology.

### Lifelog Search Challenge (LSC)

LSC is VBS's sibling campaign — same interactive, live, timed philosophy, but applied to personal lifelog data (wearable camera images, biometrics, activity data) rather than video archives. Both are described as crucial live benchmarking initiatives in the same lineage, both use DRES, and there is substantial overlap in the organizing community [4][17].

---

## 13. Where to Find Things (Quick Links)

| Resource | URL |
|----------|-----|
| Official website | `https://videobrowsershowdown.org` |
| About VBS / task types | `https://videobrowsershowdown.org/about-vbs/` |
| Call for Papers (current edition) | `https://videobrowsershowdown.org/call-for-papers/` |
| Communication with DRES | `https://videobrowsershowdown.org/about-vbs/communication-with-dres/` |
| Interaction logging spec | `https://videobrowsershowdown.org/about-vbs/interaction-logging/` |
| Publications / how to cite | `https://videobrowsershowdown.org/publications-cite/` |
| Hall of Fame (all winners) | `https://videobrowsershowdown.org/hall-of-fame/` |
| DRES source | `https://github.com/dres-dev/DRES` |
| DRES client examples | `https://github.com/dres-dev/Client-Examples` |
| DRES OpenAPI spec | `https://editor.swagger.io/?url=https://raw.githubusercontent.com/dres-dev/DRES/master/doc/oas-client.json` |
| Past tasks, results, analysis papers | `https://github.com/lucaro/VBS-Archive` |
| SOMHunter (starter system) | `https://github.com/siret/somhunter` |
| vitrivr (starter stack) | `https://vitrivr.org/` |
| TransNet V2 (shot detection) | `https://github.com/soCzech/TransNetV2` |
| V3C1 analysis data | `https://github.com/klschoef/V3C1Analysis` |
| V3C1 ASR data | `https://github.com/lucaro/V3C1-ASR` |
| Marine Video Kit | `https://mvk.hkustvgd.com` |

---

## 14. References

**Primary source**

- [1] `VBSInfo.docx` — user-provided source document, compiled from the official VBS website (About VBS; VBS Evaluation Server and Data; Existing Data, Tools and Tasks; Communication with DRES).

**Official VBS resources**

- [2] *VBS-Archive: Archive of Tasks and Results of the Video Browser Showdown.* L. Rossetto (maintainer). https://github.com/lucaro/VBS-Archive — edition history table, winners, dataset per year, analysis-paper index.
- [8] *Hall of Fame.* Video Browser Showdown. https://videobrowsershowdown.org/hall-of-fame/ — winners and award categories, 2012–2026.
- [13] *Call for Papers (VBS 2027).* Video Browser Showdown. https://videobrowsershowdown.org/call-for-papers/ — current rules, restrictions, dataset specifications, submission requirements.
- [18] *Interaction Logging.* Video Browser Showdown. https://videobrowsershowdown.org/about-vbs/interaction-logging/

**Competition analysis papers (the annual joint publications)**

- [3] K. Schoeffmann et al. *The Video Browser Showdown: a live evaluation of interactive video search tools.* International Journal of Multimedia Information Retrieval, 2013. DOI: 10.1007/s13735-013-0050-8
- [4] L. Vadicamo, R. Arnold, W. Bailer, F. Carrara, C. Gurrin, N. Hezel, X. Li, J. Lokoč, S. Lubos, Z. Ma, N. Messina, T.-N. Nguyen, L. Peška, L. Rossetto, L. Sauter, K. Schöffmann, F. Spiess, M.-T. Tran, S. Vrochidis. *Evaluating Performance and Trends in Interactive Video Retrieval: Insights from the 12th VBS Competition.* IEEE Access, 2024. DOI: 10.1109/ACCESS.2024.3405638 — **the single most useful reference**; source for AVS scoring, system capabilities, log analysis, and future-format recommendations.
- [6] K. Schoeffmann. *A User-Centric Media Retrieval Competition: The Video Browser Showdown 2012–2014.* IEEE MultiMedia, 2014. DOI: 10.1109/MMUL.2014.56
- [7] K. Schoeffmann, W. Bailer, C. Gurrin, G. Awad, J. Lokoč. *10 Years of Video Browser Showdown.* Proc. 2nd ACM Int. Conf. on Multimedia in Asia, 2021. DOI: 10.1145/3444685.3450215
- [9] K. Schoeffmann. *Video Browser Showdown 2012–2019: A Review.* CBMI 2019. DOI: 10.1109/CBMI.2019.8877397
- [10] J. Lokoč et al. *On Influential Trends in Interactive Video Retrieval: Video Browser Showdown 2015–2017.* IEEE Trans. Multimedia, 2018. DOI: 10.1109/TMM.2018.2830110
- [12] S. Heller et al. *Interactive video retrieval evaluation at a distance: comparing sixteen interactive video search systems in a remote setting at the 10th Video Browser Showdown.* IJMIR, 2022. DOI: 10.1007/s13735-021-00225-2
- [14] K. Schall, N. Hezel, K. U. Barthel, K. Jung et al. *Interactive multimodal video search: an extended post-evaluation for the VBS 2022 competition.* IJMIR, 2024. DOI: 10.1007/s13735-024-00325-9 — source of the explicit KIS scoring formula.
- Also relevant: L. Rossetto et al., *Interactive Video Retrieval in the Age of Deep Learning — Detailed Evaluation of VBS 2019*, IEEE TMM, DOI: 10.1109/TMM.2020.2980944; J. Lokoč et al., *Is the Reign of Interactive Search Eternal? Findings from VBS 2020*, ACM TOMM, DOI: 10.1145/3445031; J. Lokoč et al., *Interactive video retrieval in the age of effective joint embedding deep models: lessons from the 11th VBS*, Multimedia Systems, DOI: 10.1007/s00530-023-01143-5.

**Evaluation infrastructure**

- [11] L. Rossetto, R. Gasser, L. Sauter, A. Bernstein, H. Schuldt. *A System for Interactive Multimedia Retrieval Evaluations.* MMM 2021, pp. 385–390. DOI: 10.1007/978-3-030-67835-7_33 — the original DRES paper.
- [15] *DRES: Distributed Retrieval Evaluation Server.* https://github.com/dres-dev/DRES
- [16] L. Sauter, R. Gasser, A. Bernstein, H. Schuldt, L. Rossetto. *An Asynchronous Scheme for the Distributed Evaluation of Interactive Multimedia Retrieval.* IMuR '22. DOI: 10.1145/3552467.3554797
- [17] L. Sauter, R. Gasser, H. Schuldt, A. Bernstein, L. Rossetto. *Performance Evaluation in Multimedia Retrieval.* ACM TOMM, 2024. DOI: 10.1145/3678881 — **the currently recommended DRES citation.**

**Datasets**

- [5] Figures as given in [1] and [13], consistent across sources.
- [19] L. Rossetto, H. Schuldt, G. Awad, A. A. Butt. *V3C — A Research Video Collection.* MMM 2019, pp. 349–360. DOI: 10.1007/978-3-030-05710-7_29
- [20] Marine Video Kit (MVK). DOI: 10.48550/arXiv.2209.11518; MMM 2023 version DOI: 10.1007/978-3-031-27077-2_42
- [21] GynSurg (laparoscopic gynecology, formerly LHE75). DOI: 10.1145/3746027.3758267
- V3C2 analysis data description: https://arxiv.org/abs/2105.01475

**Tools**

- T. Souček, J. Lokoč. *TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection.* arXiv:2008.04838. https://github.com/soCzech/TransNetV2
- M. Kratochvíl, P. Veselý, F. Mejzlík, J. Lokoč. *SOM-Hunter: Video Browsing with Relevance-to-SOM Feedback Loop.* MMM 2020. DOI: 10.1007/978-3-030-37734-2_71
- vitrivr multimedia retrieval stack: https://vitrivr.org/

---

*Compiled July 2026. Competition details — especially dataset composition, task mix, and session structure — change year to year; always check the current Call for Papers before relying on rules described here.*