## Article interactivity
- [x] Click-to-expand nearest neighbors on all plots (click word → sprout top-5 neighbors)
  - Clicking a word in 2D/1D plots adds its top-5 nearest neighbors to the data, recomputes MDS, and animates the transition. Neighbors are styled distinctly (gray, smaller text) with dashed lines to their parent. Recursive: clicking neighbors expands them too.
- [ ] Adaptive neighbor count — avoid adding words that are off-topic for the current plot
  - Option A: **Relevance filter by distance.** Only add a neighbor if its average distance to the current word set is below a threshold (e.g., below the median pairwise distance). Words far from everything are likely off-topic.
  - Option B: **Subspace projection.** If the plot's words define a low-rank subspace, only add neighbors whose projection onto that subspace explains a large fraction of their variance. Orthogonal neighbors won't be informative.
  - Option C: **Similarity dropoff.** Instead of always adding 5, look at the similarity scores and stop at a gap. If top-3 are close but #4 drops off, only add 3.
  - Option D: **MDS variance impact.** Tentatively add neighbors, check how much MDS variance explained drops. If it drops a lot, the new words introduce dimensions the plot can't represent — add fewer.
- [ ] Animated steering transition (play button, words slide from original to steered with trails)
- [ ] Build-your-own word groups explorer (text area → instant MDS plot)

## Content
- [ ] Show subspaces in addition to "gender" like "size"
- [x] Add math for MDS and subspace identification to the article.
  - Added step-by-step derivations for MDS (double centering, eigendecomposition, coordinate extraction) and subspace identification (difference vectors, second moment matrix, SVD). Includes dimensionality/rank discussion and centering explanation.

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
- [ ] Animate changes from 1 -> 2 -> 3 dimensions

- [ ] can we show the subspace being identified from the group's vector, then
  their differences, followed by an (linear) adjustment, all as a
  smooth/informative/instructive animation?


## Style
- [x] Content is not correctly centered
  - Fixed alignment of equations (wrapped in `<p>`), analogy input (margin fix), cooccurrence image (changed to `<figure>`), and plots (48px left margin on plot-container). All elements now align with Distill text column.
- [ ] Remove the "waiting for review" thing (not publishing on Distill)
  - Could not find this banner in headless testing. May be browser-specific or intermittent.
- [x] Pick more attractive colors (the specific red and blue colors we have are yucky)
  - Switched to Tableau-inspired muted palette: steel blue (#5778a4) and warm orange (#e49444).
- [ ] Initial zoom on the hero plot could be increased on some platforms. Making it big is good.
- [x] Reorder gendered pairs to be feminine first.
  - All pairs now feminine first (woman/man, queen/king, etc.)

## Polish
- [x] Fix plot horizontal alignment with Distill column
  - Used Distill's `l-body-outset` class + 48px left margin to align plots with text.
- [ ] Improve 2D arrow aesthetics
- [x] Auto-orbit all 3D plots
  - Added controls.autoRotate = true to all 3D renders. Pauses on click-drag, resumes after 3s.
- [x] Larger 3D text labels
  - Doubled canvas size (512x128) and font size (48px bold) for Three.js sprite labels.
- [x] Default to 50K vocabulary
  - Changed default from small (10K) to medium (50K). 10K was missing too many words (superlatives, gendered pairs).

## Boo-boos?
- [x] Are some superlatives missing? (e.g., soft -> softer but missing softest)
  - Re-exported binaries with `must_have` parameter. All 30 superlatives now present.
- [x] Acknowledgements should be references.
  - Converted to dt-cite references in dt-appendix.
- [x] Acknowledge David Mueller https://damueller.com who co-wrote a homework assignment about word embeddings for a machine learning course at JHU.
  - Added to acknowledgments section.
