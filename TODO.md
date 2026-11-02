## Pre-launch review (2026-10-28)

Numbers below were computed on the 50K vocabulary with a one-off Python
script that mirrors `embeddings.js` (unit-normalized vectors,
query = normalize(b − a + c), cosine top-n). The script is not in the repo
yet. Smoke tests: 35/35 pass.

### Needs a decision from timv (undisclosed choices that change results)

- [ ] **Analogies exclude the three input words.** `embeddings.js:88` passes
  `new Set([a, b, c])` to `mostSimilar`. With the inputs kept, the top answer
  for man : woman :: X is X itself for every occupation in the table (doctor,
  programmer, boss, scientist, nurse, teacher, homemaker, assistant) and for
  king. For doctor, "nurse" is second. This is the critique in Nissim, van
  Noord & van der Goot (2020), "Fair is better than sensational: man is to
  doctor as woman is to doctor". The doctor → nurse example depends on this
  choice and the article doesn't mention it. Decide how to present it.
  - Scores with nothing removed (cosine to normalize(b − a + c)):
    king 0.84, queen 0.77, monarch 0.68; doctor 0.87, nurse 0.77,
    physician 0.72; homemaker 0.91, housewife 0.74; programmer 0.85,
    educator 0.59.
  - What depends on it: all analogy plots (king, doctor before/after, "Try
    your own", explorer default): the answer words shown, the highlight and
    the parallelogram's 4th corner. Also the "Try your own" sentence and
    "Also:" list (shown after clicking ?), the occupation table, and the prose
    at `index.html:279`, `:353`, `:359`, `:366`. `:353` is false as written:
    "the nearest neighbor to doctor + (woman − man) is nurse"; it's doctor.
  - Prior work: the rule comes from Mikolov et al.'s analogy test (word2vec,
    gensim). Linzen (2016, RepEval) found that without it, the nearest word
    to a* − a + b is b in 93% of cases. Bolukbasi et al. searched for whole
    pairs (x, y) with ‖x − y‖ ≤ 1 rather than solving a : b :: c : ?.
    Nissim et al. recommend reporting the full top-N whenever a rank-N word
    is reported.
  - Decision (timv, 2026-11-02): keep the exclusion.
  - [x] Disclosed. `index.html:279` now says the answer leaves out a, b and c,
    that otherwise it's usually c (Linzen, 2016), gives king 0.84 vs queen
    0.77, and notes the rule is standard since Mikolov et al. (2013) and can
    make analogies look more striking (Nissim et al., 2020). The false doctor
    sentence (`:353`) now gives doctor 0.87, nurse 0.77, physician 0.72, and
    says the gendered association is nurse outranking physician. The explorer
    help says `?` leaves out a, b and c. A comment on `analogy()` in
    `embeddings.js` explains the rule. Linzen and Nissim et al. were added to
    References.
  - [ ] Still open: the "after steering" sentences (`:359`, `:366`) depend on
    the k decision. Showing top-3 lists with scores in the table is proposed
    but not decided.

- [ ] **The analogy plots show the wrong word as closest to the query.** In
  100-D the query is closest to the input word (king; doctor). In the MDS
  layout the plots use, the closest is the excluded-rule answer:
  - man:woman::king: queen is nearest in 2D and 3D; king isn't among the 4
    nearest in 2D.
  - man:woman::doctor: nurse is nearest in 2D and 3D; in 2D, doctor is 5.9×
    farther than nurse.
  - 2D keeps only 40–42% of the variation for these word sets.
  So the plot appears to confirm the answer that only wins because the input
  word is removed. Options: label nearby words with their 100-D similarity
  to the query (overlaps the Visual Design TODO); add a caveat; or lay out
  analogy plots in a 2D slice through the query, c and the top answer, which
  keeps those three distances exact. Recommendation (pending timv): slice +
  labels.

- [ ] **k = 10 with 10 steering pairs truncates nothing.** `index.html:495`
  calls `emb.steer(GENDER_PAIRS, 10)` with 10 pairs, so the whole span of
  the difference vectors is projected out. The text at `index.html:324` says
  truncation "focuses on the dominant directions", but no truncation happens.
  - steered(woman) = steered(man) exactly (distance 0.0, was 0.58). The same
    goes for every defining pair (she/he, girl/boy, female/male, ...).
  - So every "after steering" man : woman :: X query equals steered(X), and
    the answer is X's nearest neighbor in the steered space. The after-column
    of the analogy table and "both man→woman and woman→man yield physician"
    hold by construction.
  - Scatter eigenvalues: 1.942, 1.211, 0.693, 0.216, 0.165, 0.097, 0.088,
    0.067, 0.051, 0.023. The top 1/2/3 hold 43% / 69% / 85% of the trace.
  - Options: use k = 1–3 (Bolukbasi et al. used k = 1), or keep k = 10 and
    say plainly what it does.
  - The explorer does the same: `Math.min(10, validPairs.length)` is always
    full rank for ≤ 10 pairs.
  - Results by k (2026-11-02, 50K, input words excluded as in the article):

    |                        | before    | k=1           | k=2           | k=3           | k=10          |
    |------------------------|-----------|---------------|---------------|---------------|---------------|
    | man:woman::doctor      | nurse     | physician     | physician     | physician     | physician     |
    | man:woman::nurse       | nurses    | physician     | nurses        | nurses        | doctor        |
    | man:woman::homemaker   | housewife | schoolteacher | schoolteacher | schoolteacher | schoolteacher |
    | woman:man::doctor      | dr.       | nurse         | nurse         | nurse         | physician     |
    | woman↔man distance     | 0.579     | 0.441         | 0.429         | 0.381         | 0.000         |

    The other five occupations (programmer, boss, scientist, teacher,
    assistant) give the same answer at every k. The prose claim "both
    man→woman and woman→man yield physician" (`index.html:359`) only holds at
    k = 10. At k = 1–3, woman:man::doctor still returns "nurse".
  - Recommendation (pending timv): k = 1, say that one direction isn't
    enough, and show k = 10 as the option that collapses the pairs.

- [ ] **Arrow anchors and analogy query points are part of the MDS input.**
  `buildFromDSL` (`index.html:845`) adds two synthetic anchor points per
  direction arrow, plus the analogy query point, to the word list before MDS.
  The text says each plot is "MDS on a small, hand-picked set of words".
  Measured effect (|cos(MDS axis 1, arrow direction)|, then the 2D % explained):
  - superlatives: 0.88 → 0.94; 31.3% → 33.6%
  - size pairs: 0.59 → 0.78; 36.4% → 37.2%
  - gender pairs: 0.44 → 0.52; 38.6% → 38.3%
  Options: fit MDS on the words only and place the arrow afterwards, or
  disclose it in the text.

- [ ] **Missing words are dropped silently in the article plots.**
  `buildFromDSL` filters out-of-vocabulary words, pairs and analogies with no
  warning (`warnMissing` is only called from `flatWords`, which only the
  subspace animation uses). It also does `if (!dirVec) continue` and returns
  null when no steering pair is valid. The hero filters the same way.
  Today this only matters at 10K, where "miniature" is missing, so the size
  plot silently loses the miniature/giant pair. (Checked every word in the
  DSL constants and JS word lists: nothing missing at 50K.) The 10K file is
  the top 10K words plus 27 must-have words. The explorer does show "not
  found" visibly. Options: show a note on the plot, or fail loudly.

- [ ] **MDS can't detect negative eigenvalues.** `mds()` (`embeddings.js:193`)
  takes the SVD of the double-centered matrix B and treats singular values as
  eigenvalues. Singular values are never negative, so the `Math.max(0, ·)`
  guards (`embeddings.js:237`, `viz.js:46`, `viz.js:1215`) never fire, and a
  negative eigenvalue would come back as a positive one and count toward
  "% explained". Harmless in theory for Euclidean distances, but unchecked.
  Proposal: get the sign from uᵀBu (or use a symmetric eigensolver) and fail
  loudly if a negative eigenvalue is larger than rounding error.

### Live-site bugs

- [x] **The analogy query marker (×) froze in 3D.** The 3D rotation loop
  (`EmbeddingViz._startRotation` → `tick`) moved circles, labels, arrows and
  lines, but not `text.query-marker`, so the × stayed put while its "query"
  label and dashed line rotated (measured: 24.5 px apart, then 35.5 px one
  second later).
  - Fixed: `tick` now updates the query markers too. The × stays on its label
    (0.0 px) while rotating. Added smoke test `testQueryMarker3D`.
- [x] **An ordinal rendered as raw LaTeX.** `$i\textsuperscript{th}$` in the
  scatter-matrix paragraph showed up raw ("KaTeX parse error: Undefined
  control sequence: \textsuperscript"), because KaTeX doesn't support
  `\textsuperscript`. The project CLAUDE.md prescribed that form.
  - Fixed: now `$i$<sup>th</sup>` (the convention from timv's
    conditional-poisson-sampling memory); CLAUDE.md updated to match.
    `testMathRendering` now also fails if any prose shows a raw `$`.
- [ ] **400K button returns 404.** `glove-large.bin` is gitignored, and at
  157 MB it's over GitHub's 100 MB file limit. Remove the button or host the
  file elsewhere.
- [ ] **Phone layout (checked at 390 px):**
  - The page scrolls sideways (scrollWidth 610 px). The profession bar chart
    has a fixed width (~590 px, `index.html:1251`); display equations may
    also overflow (no `overflow-x` on `.katex-display`).
  - The eigenvalue bars sit on top of plot points.
  - [x] The hero was blurry on high-DPI screens.
    - Fixed: added `renderer.setPixelRatio(window.devicePixelRatio)` in
      `renderHero3D`. Checked in headless Chrome: at 390 px @2x the canvas now
      renders at 708×1000 for a 354×500 area; desktop @1x is unchanged.
  - The hero still cuts off words at the edges. Its legend is hidden on narrow
    screens, so it has no explanation there.
  - Justified text leaves large gaps between words on narrow screens.
- [ ] **`lsi.html` is deployed but unlinked and unstyled.** It loads
  `css/blog.css`, which returns a 404 on Pages. Remove it from the deploy, or
  fix the CSS path and link it.
- [ ] Remove the draft banner (last step before launch).
- [ ] **The loading overlay hides the whole article until the vectors
  arrive.** `#loading-overlay` is a fixed full-screen layer, so readers can't
  read any prose until the 20 MB file has downloaded (2.2 s locally; much
  longer on a slow connection). Consider rendering the prose immediately and
  showing per-plot placeholders instead.
- [ ] **Old animations keep running after a vocabulary switch** (from
  reading the code; not measured). `renderHero3D` removes the old canvas but
  never calls the old hero's `destroy()`, so its animation loop and WebGL
  context stay alive. `buildVisualizations` resets `vizInstances = []` without
  stopping the old plots' rotation loops. A switch from 50K to 10K was checked
  in a headless browser: it shows one hero canvas and one SVG per plot, with
  no errors.

### Smaller items

- [ ] The doctor "after" plot's edit box shows the same plot definition as
  "before" (`DSL_DOCTOR_AFTER`), with no sign that it uses steered vectors.
  Consider adding the `~` line so the definition says what the plot does.
- [ ] Gender is defined by four different pair sets: 10 `GENDER_PAIRS` for
  steering, 14 `GENDERED_WORDS` for the arrow plot, the first 8 of those for
  the animation, and 3 in the prose example. Unify them or explain why they
  differ.
- [ ] The profession bar chart and steering plot include wife, husband,
  mother and father, which aren't occupations.
- [ ] Label collisions in the superlatives plot (strong/short/stronger/shorter
  overlap). Label collisions were marked [SKIP] under Visual Design; noting
  again only because they get much worse on phones.
- [ ] Unmeasured claim at `index.html:324`: removing 10 of 100 dimensions
  "leaving the remaining structure intact". Could be measured, e.g. as
  nearest-neighbor overlap for non-gendered words before and after steering.
- [ ] **The `~ label:` form from the DSL spec silently drops the first pair.**
  `~ steer: she - he, woman - man` parses "steer: she" as a word, and the
  pair is then filtered out. The article's own DSL constants don't use a
  label, but the spec below does. Hyphenated words (e.g. "x-ray") also break
  in any line that contains " - ".
- [ ] Dead code: `render3D` (the old Three.js inline 3D), `render1D` and
  `renderSteering2D` in `viz.js` are exported but never called. The
  click-to-expand handler is a stub that returns null, with the original in a
  comment, and four click-to-expand tests are commented out in `test.mjs`.
  Delete it, or keep it for re-enabling?
- [ ] Test gaps: nothing covers phone width, vocabulary switching, steering
  results, the 400K button, or the hardcoded prose claims. (The prose claims
  hold at both 10K and 50K: king → queen, doctor → nurse, and doctor →
  physician after steering.)
- [x] References: BERT was cited as arXiv:1810.04805, but it was published at
  NAACL 2019.
  - Fixed: the References entry now says "In *NAACL*", and both BERT links
    (margin note and References) point to the ACL Anthology page
    (https://aclanthology.org/N19-1423/).
  - Same for the other papers with free published versions: GloVe
    (D14-1162), Zhao et al. (D18-1521) and Ravfogel et al.
    (2020.acl-main.647) now link to the ACL Anthology, and Bolukbasi et al.
    to the NeurIPS proceedings. Mikolov et al. (arXiv only) and Harris (DOI)
    are unchanged; the "GloVe" name still links to the project page.
- [ ] If the input-word exclusion gets discussed, add Nissim et al. (2020,
  *Computational Linguistics*) to the references.
- [ ] The co-occurrence figure is taken from the CS224U slides and linked as
  the source. Check whether reusing it is fine, or redraw it.
- [ ] No social-preview metadata (`og:title`, `og:image`) and no publication
  date on the page. Match whatever the blog does.
- [ ] **Not yet reviewed:** most of `viz.js` (2.5K lines), including the
  subspace-animation math, the 2D→3D view search, and the steering-plot
  rendering. Also no desktop visual pass beyond the smoke tests, and no
  browser other than headless Chrome.
- [ ] The "far more aligned than random" claim (`index.html:308`) holds:
  woman−man vs she−he 0.64, woman−man vs queen−king 0.44, she−he vs
  queen−king 0.58. Random word-pair differences (top 20K words, 5000 draws):
  mean 0.00, 99th percentile 0.31. Consider putting these numbers in the text.
- [ ] **Stale entries below** (each checked against the current code; not yet
  edited):
  - Questions / aspect ratio: "See checklist below". There is no such
    checklist.
  - Questions / truncation: wrong answer (correction added in place).
  - Interactivity / steering transition: "Added `renderSteering2D`". That
    function is no longer called; `SteeringViz` replaced it.
  - Interactivity / explorer: "Click-to-expand ... all work". The feature is
    disabled.
  - Interactivity / parallelogram: "in `buildAnalogyViz`". That function was
    removed in the DSL refactor; it's `buildFromDSL` now.
  - Interactivity / eigenvalue bars: the one-time fading "click bars" hint is
    gone. Only a hover tooltip remains ("Click to switch dimensions").
  - Interactivity / profession ranking: says horizontal orange bars, labels on
    the left, values on the right. Now it's vertical steel-blue bars with
    rotated labels, and values appear only on hover.
  - Content / MDS math: "step-by-step derivations (double centering, ...)".
    These were later replaced by a short classical-MDS paragraph with no
    derivation.
  - Content / Discussion and Boo-boos / acknowledgements: "dt-cite",
    "dt-appendix". Distill is gone; there's a plain References list now.
  - Math review / analogy explanation: "rewrote as bias cancellation ...
    log co-occurrence ratios". The current text has neither.
  - Polish / alignment: "Used Distill's `l-body-outset`". It's custom CSS now.
  - Polish / click-to-expand in all dimensions, neighbors persist, neighbor
    styling in 1D/3D: the feature is disabled, and the 3D part (Three.js
    raycasting, spheres) refers to the dead `render3D`.
  - Polish / auto-orbit (`controls.autoRotate`) and larger 3D labels
    (Three.js sprites): inline 3D is D3 SVG now; Three.js is only used for
    the hero.
  - Technical polish / opening paragraph: describes the "queen closer to king
    than bicycle" opening. The article is back to the dictionary paragraph
    (commit 2003639).
  - Technical polish / Large (400K): "Enabled". It works locally only and
    returns 404 on the live site.
  - Plot DSL spec: says `= link` but the code uses `= connect`. Says
    `~ steer: ...` but a label isn't supported (see above). The "Doctor after
    steering" spec has a `~` line, but the code passes `emb: steered`.
  - Plot DSL / "Add syntax help to the explorer": partly done. The article
    lists `word - word`, `?` and `>`, but not `~` or `=`.
- [ ] Smoke tests don't check the prose claims (nurse, physician, the table),
  which are hardcoded while the results are computed live.

## Questions:

- [x] Is there a good reason to use Three.js for the hero visualization? Yes — keep it.

- crtical assessment of the fidelty of the math and code.

- [x] do we truncate the svd in subspace identification (the text suggests that we do)?  Why?
  - Yes. `steer(pairs, K=10)` keeps top-K eigenvectors. The scatter matrix has rank ≤ number of pairs, so higher components are noise. K=10 captures the dominant concept directions. This is a parameter the article should discuss (see "No discussion of how k is chosen" below).
  - Correction (2026-10-28): with 10 pairs and K=10, nothing is truncated.
    See "k = 10 with 10 steering pairs truncates nothing" under Pre-launch review.

- [x] should we set the aspect ratio to be more equal?  does that make any sense?
  - Yes. Equal aspect (1 data-unit = same pixels on both axes) is correct for MDS since it preserves distances. Implemented as default for all plots. See checklist below.

- should we try to quantify how much of the variance is accounted for in
  directions and sterring moves?  (would putting an uncertainty cone around an
  arrow make sense?  Showing the numerical value would also be nice.)

- is there a measure we can use for how well an analogy worked?  (For example,
  the man:woman analogies after gender steering should all be mostly junk and
  that should be something we can test for ideally.)

- There are some cool examples (e.g., city:zipcode) and useful content in to
  draw from https://nlp.stanford.edu/projects/glove/

- Are there places that could use visual aids?  (An example of a missing visual
  aid is the matrix factoriization.)

- Are there places that are lacking mathematical detail or rigor?  (E.g., matrix
  factorization, and MDS explanation are a bit weak.)

- AI explanation of surprising analogies — currently an "explain?" link that
  opens Perplexity with a pre-filled query. Inline display would be better but
  blocked by CORS (can't fetch from perplexity.ai in the browser). Options:
  - **Link out to Perplexity** (current) — works, no server needed, but leaves the page.
  - **Proxy server** — would allow inline display but breaks "runs entirely in the browser."
  - **Perplexity API** — requires API key, not free, can't embed key in client JS.
  - **iframe** — Perplexity blocks framing (X-Frame-Options).
  - Could revisit if we ever add a backend, or if a free chat API with CORS support appears.


## Remember this

- Please make sure that when transition between 1d <-> 2d <-> 3d plots that
  there isn't an unncessary change in zooming.  We worked really hard to getting
  this detail right in previous iterations - I would hate for that to get lost
  in this change.

  [SKIP] TODO [2026-03-31 Tue] This issue reemerged when we switched to using equal
      aspect ratio.


## Layout

- [ ] Show an embedding visualization as early as possible — ideally right under
      the first paragraph, before the co-occurrence matrix explanation. Give the
      reader something concrete before the theory. Options:
      - Show raw embedding vectors for a few words (e.g., cat, dog, king, queen)
        as 100-dimensional lists of numbers. Then show that cat/dog are close
        (high similarity) while cat/king are far (low similarity). Makes
        "a list of numbers" tangible before jumping to co-occurrence matrices.
      - An MDS plot of a small word group early on.
      - Both: raw vectors first, then the plot.

## Math Rendering

- [ ] Switch from KaTeX to MathJax to match the blog template (which uses MathJax 3).

## Visual Design

- [ ] Finetune the appearance of the query point in analogy plots.
      Currently rendered as a × marker with italic label in highlight green
      (#5e8c61), 0.8 opacity, 16px bold. The circle is hidden (r=0) and a
      dashed line connects from the base word (c) to the query point.
      Also annotate the distances from the query point to nearby answer
      words with their actual cosine distances.

- [SKIP] I don't want to see unncessary labels collisions.  Please improve the
      layout, as it doesn't appear to try any alternative label placements to
      avoid collisions.


## Article interactivity

- [x] Click-to-expand nearest neighbors — **disabled**
  - Infrastructure remains in viz.js (`_makeOnClick`, neighborWords/neighborLinks styling in render2D/1D/3D). To re-enable: change `_makeOnClick()` to return the handler instead of null.
  - Disabled because: expanded neighbors were often off-topic, MDS quality degraded as words were added, labels got crowded fast.
  - If re-enabled, consider adaptive neighbor count (relevance filter, similarity dropoff, or MDS variance impact).

- [x] Animated steering transition (play button, words slide from original to steered with trails)
  - Added `renderSteering2D` in viz.js. Computes joint MDS over original+steered positions, then D3 transitions animate words from original to steered with ghost dots and trails. Steer/Reset buttons. Replaces the two separate before/after gendered pair plots.

- [x] Build-your-own word groups explorer (text area → instant MDS plot)
  - Added "Explore" section at bottom with textarea and MDS viz. Live update on input (debounced 500ms).
  - `word - word` syntax defines steering pairs; Steer button appears when pairs are present.
  - Words on a line connected by arrows. Ghost arrows + ghost dots show original positions during steering.
  - Click-to-expand, pan/zoom, dimension switching all work.

- [x] Draw the parallelogram in analogy visualizations: show a→b and c→answer
  as parallel arrows to make the vector arithmetic visible.
  - Added dashed crossGroupLines connecting a→c and b→answer in buildAnalogyViz.

- [x] Make the eigenvalue bars more discoverable — add a tooltip or small
  "click to change dimensions" hint on first appearance.
  - Added SVG tooltip explaining eigenvalues and click-to-switch. Added one-time "click bars to change dimensions" hint that fades out after 4s.

- [x] Replace the plain `<ol>` profession ranking with a horizontal bar chart
  showing shift magnitude, consistent with the rest of the visual style.
  - D3 horizontal bar chart with orange bars, word labels on left, shift values on right.


## Content

- [x] Show subspaces in addition to "gender" like "size"
  - Added SIZE_PAIRS (small/large, tiny/huge, etc.) with SVD-derived "size direction" arrow. Plot inserted between gender pairs and the subspace math.

- [x] Add math for MDS and subspace identification to the article.

  - Added step-by-step derivations for MDS (double centering,
    eigendecomposition, coordinate extraction) and subspace identification
    (difference vectors, second moment matrix, SVD). Includes
    dimensionality/rank discussion and centering explanation.

- [x] adjust the explanation to indicate that there are other ways one might
  come up with embeddings for words that are not based on co-occurence
  statistics (word2vec skip-gram/CBOW, contextual embeddings like BERT).
  Include citations. Please mark edits in purple.
  - Added purple paragraph mentioning word2vec (skip-gram/CBOW) and BERT with citations.

- [x] the idea of a matrix factorization is missing: there should be a target
  matrix and them a way of reconstructing it from a lower dimensional
  representation.  In this case, the log co-occurence matrix is being
  reconstructed from the 100-dimensional glove embeddings.
  - Added paragraph explaining GloVe as log co-occurrence matrix factorization with formula.

- [x] Add intuition for *why* analogies work — connect the log-bilinear GloVe
  objective to the linearity of the resulting space.
  - Added paragraph after the analogy formula connecting log co-occurrence ratios to vector offsets.

- [x] Add a concluding section after Results: discuss what the results mean,
  limitations (projecting out gender may damage useful information), and
  connections to more recent debiasing work.
  - Added Discussion section covering limitations and citing Zhao et al. and Ravfogel et al. (INLP).
  - Fixed: Zhao et al. and Ravfogel et al. were plain text — converted to proper dt-cite with bibliography entries.

- [x] Expand the "Identifying Subspaces" section with a worked example: show
  2-3 gender pair difference vectors pointing in roughly the same direction
  before jumping to the covariance matrix / SVD formalism.
  - Added concrete example with woman-man, she-he, queen-king difference vectors.

- [x] Clarify "steering" vs "debiasing" terminology — note that Bolukbasi et al.
  call this "debiasing" so readers searching for related work can find it.
  - Added parenthetical in "Steering by Subspace Projection" section.

- [x] Explain *why* re-normalization happens after projection in the steering
  formula (cosine similarity is the standard metric for these vectors).
  - Added sentence explaining unit-length requirement for cosine similarity.


## Plot DSL

- [x] Implement the plot DSL: a plain-text specification language that can generate every static plot in
the article and serve as the explorer's input format.
  - Added `parseDSL`, `computeDirectionVec`, `buildFromDSL` functions. All 11 article plots converted to DSL string constants. Edit button (✎) on each plot for live DSL editing. Removed `buildVizWithDirection` and `buildAnalogyViz`.

### Primitives

The language has five line types, distinguished by a leading sigil:

```
(bare line)   word groups — arrows connect consecutive words on each line
>             direction arrow — overlay a semantic direction on the plot
=             cross-links — connect corresponding words across groups by position
~             steer — project out a subspace, show ghost dots + trails
?             analogy — compute vector arithmetic, draw parallelogram
```

### Syntax

**Word groups** (no sigil). Each line is a group. Arrows connect consecutive
words within a group. Multiple groups appear as distinct chains/clusters.

```
poor poorer poorest
rich richer richest
fast faster fastest
```

**Direction arrow** (`>`). Computes a direction in 100-D and overlays a dashed
red arrow spanning the point cloud. Two modes:

- Pair-SVD (default): pairs separated by `-`. Top eigenvector of the scatter
  matrix of pair differences. Label follows the colon.
  ```
  > degree: poor - poorest, rich - richest, fast - fastest
  ```

- PCA: just a word sequence (no `-`). Top principal component of those vectors,
  oriented first→last.
  ```
  > counting: one two three four five six seven eight nine
  ```

Multiple direction lines = multiple arrows (as in linked numbers).

**Cross-links** (`=`). Connect the $i$-th word of each group by dashed lines.
Requires groups to have the same length.

```
one two three
1 2 3
= link
```

**Steer** (`~`). Define pairs for subspace projection. Produces a Steer/Reset
button; clicking it animates words from original to steered positions with ghost
dots and trails.

```
~ steer: she - he, woman - man, her - him, herself - himself
```

When `~` is present, word groups are plotted using joint MDS over
[steered, original] positions so ghosts and steered dots share a coordinate
frame.

**Analogy** (`?`). Computes $a : b :: c \to ?$ via vector arithmetic. Adds result
words to the plot and draws the parallelogram (a→b, c→answer as arrows;
a↔c, b↔answer as dashed cross-links).

```
? man : woman :: king
```

### Every article plot as DSL

**Superlatives:**
```
poor poorer poorest
rich richer richest
short shorter shortest
slow slower slowest
fast faster fastest
soft softer softest
strong stronger strongest
mean meaner meanest
dark darker darkest
smart smarter smartest
> degree: poor - poorest, rich - richest, short - shortest, slow - slowest, fast - fastest, soft - softest, strong - strongest, mean - meanest, dark - darkest, smart - smartest
```

**Digits:**
```
1 2 3 4 5 6 7 8 9
> counting: 1 2 3 4 5 6 7 8 9
```

**Number words:**
```
one two three four five six seven eight nine
> counting: one two three four five six seven eight nine
```

**Linked numbers:**
```
one 1
two 2
three 3
four 4
five 5
six 6
seven 7
eight 8
nine 9
= link
> word vs digit: one - 1, two - 2, three - 3, four - 4, five - 5, six - 6, seven - 7, eight - 8, nine - 9
> counting: one two three four five six seven eight nine
```

**Gender pairs:**
```
woman man
queen king
actress actor
girl boy
mom dad
mother father
sister brother
aunt uncle
heiress heir
duchess duke
niece nephew
madame sir
female male
feminine masculine
> gender: woman - man, queen - king, actress - actor, girl - boy, mom - dad, mother - father, sister - brother, aunt - uncle, heiress - heir, duchess - duke, niece - nephew, madame - sir, female - male, feminine - masculine
```

**Size pairs:**
```
small large
tiny huge
little big
narrow wide
short tall
thin thick
shallow deep
minor major
miniature giant
> size: small - large, tiny - huge, little - big, narrow - wide, short - tall, thin - thick, shallow - deep, minor - major, miniature - giant
```

**Analogy (king):**
```
? man : woman :: king
```

**Gendered steering:**
```
woman man
queen king
actress actor
girl boy
mom dad
mother father
sister brother
aunt uncle
heiress heir
duchess duke
niece nephew
madame sir
female male
feminine masculine
~ steer: she - he, woman - man, herself - himself, her - him, hers - his, gal - guy, girl - boy, girls - boys, female - male, females - males
```

**Doctor before steering:**
```
? man : woman :: doctor
```

**Doctor after steering:**
```
? man : woman :: doctor
~ steer: she - he, woman - man, herself - himself, her - him, hers - his, gal - guy, girl - boy, girls - boys, female - male, females - males
```

**Profession steering:**
```
caretaker homemaker doctor nurse programmer teacher wife husband soldier salesperson analyst therapist trainer instructor ceo assistant telemarketer bartender clerk designer father mother scientist manager boss employee
~ steer: she - he, woman - man, herself - himself, her - him, hers - his, gal - guy, girl - boy, girls - boys, female - male, females - males
```

### Open questions

- **Subspace animation**: The 5-step walkthrough (pairs → differences →
  direction → projection → steer) is a distinct visualization mode, not just a
  static plot. Should the DSL trigger it (e.g., `~ animate: ...`), or leave it
  as a separate component?

- **Hero viz**: Three.js 3D scene with auto-orbit. Probably stays as its own
  thing — it's the only WebGL component.

- **Analogy table**: The before/after occupation table is data, not a plot. Could
  be generated from the DSL (multiple `?` lines + `~ steer`), but might be
  cleaner as a separate widget.

- **Neighbor count for analogies**: `? man : woman :: king` currently shows 10
  results. Should the count be configurable (`? man : woman :: king @ 5`)?

- **Direction arrow from steer pairs**: When `~` is present, should we
  automatically show a direction arrow for the steered-out subspace? Currently
  the gendered steering plot doesn't show one, but it would be informative.

- **Multiple embeddings**: The doctor-after plot uses the steered embedding for
  the analogy. The `~ steer` line defines which embedding to steer. Should the
  DSL support running the analogy on the steered embedding explicitly?

### Implementation plan

- [x] Write a parser: text → structured spec (word groups, directions, steer pairs,
   analogies, neighbor expansions, cross-links).
   - Added `parseDSL(text)` returning `{ displayGroups, steerPairs, analogies, directions, connectGroups }`.

- [x] Write a builder: spec → EmbeddingViz or SteeringViz config. This replaces
   `buildVizWithDirection`, `buildAnalogyViz`, and the manual joint-MDS code.
   - Added `buildFromDSL(dslText, plotEl, eigenEl, options)` and `computeDirectionVec()`.
   - Removed `buildAnalogyViz` and `buildVizWithDirection`.

- [x] Rewrite `buildVisualizations()` to use the DSL (each plot is a string
   constant parsed by the same code path).
   - All 11 article plots converted to DSL string constants. Hero, subspace animation, analogy table, and bar chart kept as-is.

- [x] Replace the explorer's parser with the same DSL parser.
   - Explorer now uses `parseDSL` and `buildFromDSL`.

- [x] Convert custom analogy input to use `buildFromDSL`.
   - `setupAnalogyInput` constructs a DSL string `? a : b :: c` and calls `buildFromDSL`.

- [x] Add edit button to each DSL-built plot.
   - Subtle pencil icon in top-right of figure-row, visible on hover. Toggles a textarea with the DSL, live re-rendering on change (debounced 500ms).

- [ ] Add syntax help to the explorer (tooltip or collapsible reference).


## AWESOMENESS

- [x] Animate changes from 1 -> 2 -> 3 dimensions
  - Unified all rendering to D3/SVG — no more Three.js for inline plots.
  - 3D is projected via rotation matrix, auto-rotates with requestAnimationFrame.
  - All dimension switches are smooth D3 transitions (same SVG, same elements).
  - 1D↔2D: y-coords collapse/spread. 2D↔3D: points slide to projected positions.
  - Three.js kept only for hero visualization.

- [x] can we show the subspace being identified from the group's vector, then
  their differences, followed by an (linear) adjustment, all as a
  smooth/informative/instructive animation?
  Added `renderSubspaceAnimation` in viz.js with 5-step walkthrough:
  Pairs → Differences (translated to centroid) → Direction → Projection → Steer.
  Placed after the SVD math, before "Steering by Subspace Projection."


- [x] the 3d steering method step-by-step should project all points onto a
  plane, just like in 2d this project onto a line.
  - Added semi-transparent perpendicular plane (polygon) in 3D mode. Computed
    via cross products of direction vector to get two orthogonal basis vectors.
    Plane appears at step 3 (projection), fades at step 4 (steer).


- [ ] In the steering animations, it's a little hard to figure out what exactly
  is going on.  I wonder if there is some way to use the step-by-step animation
  here.  Maybe we can show the influence of the gendered pairs is a better way,
  e.g., are all points moving in a common gender direction, can we actually see
  it in the MDS plot?  How do we interpret the steering movements?


## Style


- [x] animations between 2d and 3d should find the closet 3d view to the current
  2d view.  There appears to be an unnecessarily large change.
  - Coarse-to-fine rotation search (72 angles × 15 tilts, then ±3° refinement) with Procrustes-style optimal scaling.
  - Fixed zoom jump: all dims now normalized to [-1,1] with fixed domain [-1.15,1.15], so viewport doesn't refit during transitions.


- [x] I cant rotate the points along all axes in 3d.
  - Added vertical drag (tilt angle) to the 3D projected view. Horizontal drag = rotation, vertical drag = tilt (clamped ±90°).


## Polish

- [x] co-occurence matrix symbol is used without introduction.
  - Introduced $\mathbf{X}$ and $\mathbf{X}_{ij}$ inline before the GloVe equation.

- [x] Fix plot horizontal alignment with Distill column
  - Used Distill's `l-body-outset` class + 48px left margin to align plots with text.

- [x] Pan + zoom on all 2D and 1D plots
  - Added d3.zoom to all SVG plots (render2D, render1D, renderSteering2D). Scroll to zoom (0.5x-5x), drag background to pan. Zoom filter excludes clicks on circles/text so click-to-expand still works.

- [x] Click-to-expand works in all dimensions (1D, 2D, 3D)
  - 2D: D3 click handlers on circles and text labels.
  - 3D: Three.js raycasting on sphere meshes with drag-vs-click detection.
  - 1D: D3 click handlers on circles and text labels with stopPropagation.

  - [x] Expanded neighbors persist across dimension switches.

- [x] Neighbor styling (gray dots, dashed links, smaller labels) propagates to 1D and 3D
  - 1D: added neighborWords/neighborLinks support, gray fills, dashed lines.
  - 3D: gray sphere color, lighter label text, LineDashedMaterial links.
  - Tested: 2D→1D switch preserves gray colors and dashed links.

- [x] Auto-orbit all 3D plots
  - Added controls.autoRotate = true to all 3D renders. Pauses on click-drag, resumes after 3s.

- [x] Larger 3D text labels
  - Doubled canvas size (512x128) and font size (48px bold) for Three.js sprite labels.

- [x] Default to 50K vocabulary
  - Changed default from small (10K) to medium (50K). 10K was missing too many words (superlatives, gendered pairs).

- [x] Remove blog.css cache-buster script from index.html before deploying to production.

## Technical polish

- [x] Define a `\defeq` macro for definitional equalities (as opposed to plain
  `=`). Use KaTeX's `macros` option in the `renderMathInElement` call — not
  `\newcommand` in the document body.
  - Already done: macro defined in `renderMathInElement` call, used for scatter matrix, steering formula.

- [x] some places write w_i^\top w_j \approx \log X_{ij}, but that isn't quite
  right as there is b_i and b_j.
  - Fixed: the formal equation (line 185) includes bias terms. Informal prose ("dot products approximate log counts") is acceptable as a gloss.

- [x] Improve co-occurrence image accessibility: add descriptive alt text, or
  replace the external PNG with an inline SVG/table.
  - Added detailed alt text describing the matrix structure.

- [x] Make plot dimensions responsive — currently hardcoded at 620x450px,
  overflows on narrow screens.
  - Added getResponsiveWidth() helper; all render functions read container clientWidth. Height uses 0.72 aspect ratio. Capped at 900px, fallback 620px.

- [x] Add error handling for failed binary loads (fetch error, 404, offline) —
  show a user-visible message instead of leaving the loading overlay up forever.
  - try/catch around loadEmbeddings; on failure shows red progress bar + error message + suggestion to try smaller vocab.

- [x] Clean up the "Large (400K)" vocab option — buttons are commented out in
  HTML but config and data file exist. Either enable it or remove the dead code.
  - Enabled: added Large (400K) buttons to both loading and inline vocab selectors.

- [x] Tighten the opening paragraph — lead with something more concrete or
  surprising rather than the dictionary-definition style.
  - Replaced dictionary/embedding parallel with concrete example: "queen" is closer to "king" than "bicycle" — leads with the payoff.

- [x] Add context to the hero visualization — readers land on trails and colored
  dots before they know what embeddings or steering are.
  - Added orienting caption: "Each dot is a word; faded dots show original positions; bright dots show where they land after a gender direction is removed; trails show the shift."

- [x] Beef up the Results section — the doctor/nurse example is a single case
  stated in prose then shown in two plots. A brief table or a couple more
  occupation examples inline would make it land harder.
  - Added dynamically-generated table showing man:woman::X analogy results before/after steering for 8 occupations. Changed results highlighted in orange.

- [x] Un-purple the Discussion section — first paragraph (limitations of
  projection) should be in black; it's the main takeaway, not supplementary.
  - All purple text was removed in a previous session.

- [x] Add a transition into "Identifying Subspaces" — bridging sentence from
  analogies to subspace identification.
  - Rewrote opening: "The consistency of these analogies suggests that the embedding space contains interpretable subspaces — directions we can find systematically, not just stumble across in individual word pairs."

- [x] Add an intuitive gloss before the steering formula.
  - Added: "The idea is simple: to remove a concept, subtract each word's projection onto that subspace and renormalize."

- [x] Improve the co-occurrence figure caption — was just "source".
  - Added inline description: "A word co-occurrence matrix: each cell counts how often two words appear near each other in a corpus."

- [x] Split the GloVe paragraph — it introduces GloVe, explains the
  factorization objective, and connects it to linear structure all in one purple
  block. The "which is why analogies work" payoff should be more prominent.
  - Split into two paragraphs: (1) factorization mechanics, (2) why linear structure emerges and its consequences.

- [x] Clarify the Explore section syntax.
  - Rewrote instructions: explains that words on the same line are plotted together and connected by arrows, and that `word - word` defines steering pairs.


## Math review

### Easy

- [x] Eigendecomposition uses SVD notation — changed $\boldsymbol{\Sigma}$ to $\boldsymbol{\Lambda}$
- [x] Variance-explained denominator — changed $\sum_{j=1}^n$ to $\sum_j$ (no wrong upper bound)
- [x] "Second-moment matrix" → "scatter matrix", removed the $1/2$ factor from formula and code,
      dropped the confusing "two observations with one degree of freedom" explanation
- [x] Analogy explanation — restored bias terms, rewrote as bias cancellation in differences
      yielding log co-occurrence ratios

### Medium

- [x] MDS loss function (raw stress) doesn't match the implementation (classical
      MDS via double-centering). Either show the classical formulation or note
      that the optimization view is for intuition and the code uses the
      closed-form solution
      - Removed raw stress formula. Described the goal informally, then introduced classical MDS / PCA as the closed-form solution we actually use.
- [x] The $\frac{1}{2}$ factor explanation ("two observations with one degree of
      freedom") is cryptic — rewrite with a clearer motivation or just drop the
      factor since eigenvectors are unaffected
      - Already removed in previous rewrite of scatter matrix section.
- [x] Vectors are L2-normalized on load but this is never stated — all distances
      are cosine-based, which changes the interpretation of "Euclidean distance"
      throughout
      - Added "normalized to unit length" to the GloVe description sentence.
- [x] Steering formula presents renormalization as inherent to the method, but
      it's only needed because we work on the unit sphere — note this
      - Clarified that renormalization is a consequence of working with unit vectors, not inherent to projection.

### Hard

- [SKIP] GloVe uses two separate embedding matrices ($W$ and $\tilde{W}$), not
      $WW^\top$. Intentionally simplified — the article is about embeddings and
      steering, not GloVe internals. The less GloVe-specific detail the better.
- [x] No discussion of how $k$ (subspace dimension) is chosen — the code uses
      $K=10$ out of 100 dimensions with no justification. Needs at least a
      paragraph on eigenvalue spectrum, sensitivity, and tradeoffs
      - Added tradeoff discussion: too small misses secondary directions, too large distorts embedding. Eigenvalue spectrum as diagnostic. Bolukbasi used k=1; we use k=10 to match our 10 pairs.


## Boo-boos?

- [x] Are some superlatives missing? (e.g., soft -> softer but missing softest)

  - Re-exported binaries with `must_have` parameter. All 30 superlatives now present.

  - [x] errors/warnings if words are missing to prevent issues like this.
    - Added `warnMissing()` helper; called for superlatives, digits, number words, gendered words, and professions. Logs missing words to console with labels.

- [x] Acknowledgements should be references.

  - Converted to dt-cite references in dt-appendix.

- [x] Acknowledge David Mueller https://damueller.com who co-wrote a homework assignment about word embeddings for a machine learning course at JHU.

- Added to acknowledgments section.
