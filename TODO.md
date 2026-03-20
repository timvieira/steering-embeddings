## Questions:

- Is there a good question to use Three.js kept for hero visualization?


## Article interactivity

- [x] Click-to-expand nearest neighbors on all plots (click word → sprout top-5 neighbors)
  - Clicking a word in 2D/1D plots adds its top-5 nearest neighbors to the data, recomputes MDS, and animates the transition. Neighbors are styled distinctly (gray, smaller text) with dashed lines to their parent. Recursive: clicking neighbors expands them too.

- [ ] Adaptive neighbor count — avoid adding words that are off-topic for the current plot
  - Option A: **Relevance filter by distance.** Only add a neighbor if its average distance to the current word set is below a threshold (e.g., below the median pairwise distance). Words far from everything are likely off-topic.
  - Option B: **Subspace projection.** If the plot's words define a low-rank subspace, only add neighbors whose projection onto that subspace explains a large fraction of their variance. Orthogonal neighbors won't be informative.
  - Option C: **Similarity dropoff.** Instead of always adding 5, look at the similarity scores and stop at a gap. If top-3 are close but #4 drops off, only add 3.
  - Option D: **MDS variance impact.** Tentatively add neighbors, check how much MDS variance explained drops. If it drops a lot, the new words introduce dimensions the plot can't represent — add fewer.

- [x] Animated steering transition (play button, words slide from original to steered with trails)
  - Added `renderSteering2D` in viz.js. Computes joint MDS over original+steered positions, then D3 transitions animate words from original to steered with ghost dots and trails. Steer/Reset buttons. Replaces the two separate before/after gendered pair plots.

- [ ] Build-your-own word groups explorer (text area → instant MDS plot)

- [x] Draw the parallelogram in analogy visualizations: show a→b and c→answer
  as parallel arrows to make the vector arithmetic visible.
  - Added dashed crossGroupLines connecting a→c and b→answer in buildAnalogyViz.

- [ ] Make the eigenvalue bars more discoverable — add a tooltip or small
  "click to change dimensions" hint on first appearance.

- [x] Replace the plain `<ol>` profession ranking with a horizontal bar chart
  showing shift magnitude, consistent with the rest of the visual style.
  - D3 horizontal bar chart with orange bars, word labels on left, shift values on right.


## Content

- [ ] Show subspaces in addition to "gender" like "size"

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

- [ ] under the section "Which Words Changed Most?" show the points using a plot!


## Hero visualization

- [x] Marquee 3D scene at top with gendered pairs + professions, ghost/bright dots, trails

- [ ] Marquee moves into margin on scroll; sections activate/deactivate parts of the scene


## AWESOMENESS

- [x] Animate changes from 1 -> 2 -> 3 dimensions
  - Unified all rendering to D3/SVG — no more Three.js for inline plots.
  - 3D is projected via rotation matrix, auto-rotates with requestAnimationFrame.
  - All dimension switches are smooth D3 transitions (same SVG, same elements).
  - 1D↔2D: y-coords collapse/spread. 2D↔3D: points slide to projected positions.
  - Three.js kept only for hero visualization.

- [ ] can we show the subspace being identified from the group's vector, then
  their differences, followed by an (linear) adjustment, all as a
  smooth/informative/instructive animation?


## Style

- [x] Content is not correctly centered
  - Fixed alignment of equations (wrapped in `<p>`), analogy input (margin fix), cooccurrence image (changed to `<figure>`), and plots (48px left margin on plot-container). All elements now align with Distill text column.

- [x] Remove the "waiting for review" thing (not publishing on Distill)
  - Hidden via CSS: `dt-banner { display: none !important; }` (element selector, not class)

- [x] Pick more attractive colors (the specific red and blue colors we have are yucky)
  - Switched to Tableau-inspired muted palette: steel blue (#5778a4) and warm orange (#e49444).

- [ ] Initial zoom on the hero plot could be increased on some platforms. Making it big is good.

- [x] Reorder gendered pairs to be feminine first.
  - All pairs now feminine first (woman/man, queen/king, etc.)

- [x] move the text "3D MDS captures 42.8% of variance" over to the right of the
  plot around the [1,2,3} histogram thing.
  - Moved variance % into the eigenvalue margin widget (below the bars). Removed from plot container.

- [x] text inside visualizations constantly text selected when I interact with
  the plots; I think it would be better to make that text unselectable.
  - Added `user-select: none` to `svg.plot` in CSS.

- [ ] animations between 2d and 3d should find the closet 3d view to the current
  2d view.  There appears to be an unnecessarily large change.

- [x] I cant rotate the points along all axes in 3d.
  - Added vertical drag (tilt angle) to the 3D projected view. Horizontal drag = rotation, vertical drag = tilt (clamped ±90°).


## Polish

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

## Technical polish

- [x] Improve co-occurrence image accessibility: add descriptive alt text, or
  replace the external PNG with an inline SVG/table.
  - Added detailed alt text describing the matrix structure.

- [ ] Make plot dimensions responsive — currently hardcoded at 620x450px,
  overflows on narrow screens.

- [ ] Add error handling for failed binary loads (fetch error, 404, offline) —
  show a user-visible message instead of leaving the loading overlay up forever.

- [ ] Clean up the "Large (400K)" vocab option — buttons are commented out in
  HTML but config and data file exist. Either enable it or remove the dead code.

- [ ] Tighten the opening paragraph — lead with something more concrete or
  surprising rather than the dictionary-definition style.


## Boo-boos?

- [x] Are some superlatives missing? (e.g., soft -> softer but missing softest)

  - Re-exported binaries with `must_have` parameter. All 30 superlatives now present.

  - [ ] errors/warnings if words are missing to prevent issues like this.

- [x] Acknowledgements should be references.

  - Converted to dt-cite references in dt-appendix.

- [x] Acknowledge David Mueller https://damueller.com who co-wrote a homework assignment about word embeddings for a machine learning course at JHU.

- Added to acknowledgments section.
