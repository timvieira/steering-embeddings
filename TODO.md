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


## Content

- [ ] Show subspaces in addition to "gender" like "size"

- [x] Add math for MDS and subspace identification to the article.

  - Added step-by-step derivations for MDS (double centering,
    eigendecomposition, coordinate extraction) and subspace identification
    (difference vectors, second moment matrix, SVD). Includes
    dimensionality/rank discussion and centering explanation.


## tutorial

- [ ] adjust the explanation to indicate that there are other ways one might
  come up with embeddings for words that are not based on co-occurence
  statistics.  We should give citations for that.  Please mark edits in purple.

- [ ] the idea of a matrix factorization is missing: there should be a target
  matrix and them a way of reconstructing it from a lower dimensional
  representation.  In this case, the log co-occurence matrix is being
  reconstructed from the 100-dimensional glove embeddings.

## Hero visualization

- [x] Marquee 3D scene at top with gendered pairs + professions, ghost/bright dots, trails

- [ ] Marquee moves into margin on scroll; sections activate/deactivate parts of the scene

- [ ] Consider starting without labels, revealing them as reader scrolls

- [x] The ghost of the previous position is too faint.
  - Increased ghost dot opacity from 0.25 to 0.55 and trail opacity from 0.3 to 0.45 in renderHero3D.

- [x] What do the red/blue colors mean?
  - Added a color legend below the hero: blue circle = gendered word pairs, orange circle = occupations.

- [x] Scroll gets trapped by 3D canvas zoom.
  - Re-enabled zoom but added visible border + grab cursor on all 3D canvases so boundary is clear.

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

- [ ] move the text "3D MDS captures 42.8% of variance" over to the right of the
  plot around the [1,2,3} histogram thing.

- [ ] text inside visualizations constantly text selected when I interact with
  the plots; I think it would be better to make that text unselectable.

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

## Boo-boos?

- [x] Are some superlatives missing? (e.g., soft -> softer but missing softest)

  - Re-exported binaries with `must_have` parameter. All 30 superlatives now present.

  - [ ] errors/warnings if words are missing to prevent issues like this.

- [x] Acknowledgements should be references.

  - Converted to dt-cite references in dt-appendix.

- [x] Acknowledge David Mueller https://damueller.com who co-wrote a homework assignment about word embeddings for a machine learning course at JHU.

- Added to acknowledgments section.
