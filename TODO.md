## Questions:

- [x] crtical assessment of the fidelty of the math and code.
  - Done: see "Article review (2026-04-02)" section below.

- [ ] There are some cool examples (e.g., city:zipcode) and useful content in to
  draw from https://nlp.stanford.edu/projects/glove/

- [x] Are there places that are lacking mathematical detail or rigor?
  - Addressed in article review (2026-04-02): log factorization, scatter matrix, $k$ justification. Remaining level of detail is appropriate for an interactive explainer.


## Article review (2026-04-02)

### Mathematical issues

- [x] "Factorizing the co-occurrence matrix" vs. the log (line 183): now says
      "factorizing the log co-occurrence matrix." Also split the sentence and
      linked the citation to the PDF instead of duplicating the project URL.

- [x] Analogy framing mismatches the article's theme (line 229): reframed as
      "man is to woman as king is to queen" with equation
      $\vec{man} - \vec{woman} \approx \vec{king} - \vec{queen}$.

- [x] Scatter matrix: redundant transition (lines 276–278): removed "The scatter
      matrix is:" ending; now flows into a display equation, then prose continues.

- [x] $k = 10$ contradicts the tradeoff discussion (line 278): added sentence
      "With 10 pairs in a 100-dimensional space, this removes only 10% of the
      embedding dimensions, leaving the remaining structure intact."

- [x] "Discards noise" → "discards weaker ones."

### Clarity / structural issues

- [x] Scatter matrix paragraph: split into three paragraphs (definition +
      display equation, eigendecomposition + basis, rank/tradeoff/$k$ choice).

- [x] "Parallel subspaces" → "parallel regions" in Numbers section.

- [x] Added "Try it yourself:" transition into Explore section.

### Small writing issues

- [x] Double GloVe citation: project page link kept on "GloVe", paper PDF link
      on "(Pennington et al., 2014)".

- [x] "Explains" → "represents" in the opening.

### Narrative

- [x] Transition from Structure to Analogies: now opens with "The superlatives
      already demonstrate analogies — poor is to poorer as rich is to richer —
      and the pattern generalizes."

- [x] Discussion: added concluding sentence about geometry that encodes meaning
      being the geometry we can edit.


## Remember this

- Please make sure that when transition between 1d <-> 2d <-> 3d plots that
  there isn't an unncessary change in zooming.  We worked really hard to getting
  this detail right in previous iterations - I would hate for that to get lost
  in this change.


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

- [ ] Implement the plot DSL: a plain-text specification language that can generate every static plot in
the article and serve as the explorer's input format.

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

1. Write a parser: text → structured spec (word groups, directions, steer pairs,
   analogies, neighbor expansions, cross-links).

2. Write a builder: spec → EmbeddingViz or SteeringViz config. This replaces
   `buildVizWithDirection`, `buildAnalogyViz`, and the manual joint-MDS code.

3. Rewrite `buildVisualizations()` to use the DSL (each plot is a string
   constant parsed by the same code path).

4. Replace the explorer's parser with the same DSL parser.

5. Add syntax help to the explorer (tooltip or collapsible reference).


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
