## Article interactivity
- [ ] Click-to-expand nearest neighbors on all plots (click word → sprout top-5 neighbors)
- [ ] Animated steering transition (play button, words slide from original to steered with trails)
- [ ] Build-your-own word groups explorer (text area → instant MDS plot)

## Content
- [ ] Show subspaces in addition to "gender" like "size"
- [ ] Add math for MDS and subspace identification to the article.

## Hero visualization
- [x] Marquee 3D scene at top with gendered pairs + professions, ghost/bright dots, trails
- [ ] Marquee moves into margin on scroll; sections activate/deactivate parts of the scene
- [ ] Consider starting without labels, revealing them as reader scrolls
- [x] The ghost of the previous position is too faint.
  - Increased ghost dot opacity from 0.25 to 0.55 and trail opacity from 0.3 to 0.45 in renderHero3D.
- [x] What do the red/blue colors mean?
  - Added a color legend below the hero: blue circle = gendered word pairs, red circle = professions.
- [x] Scroll gets trapped by 3D canvas zoom.
  - Disabled scroll zoom on the hero (controls.enableZoom = false). Users can still click-drag to orbit.

## AWESOMENESS
- [ ] Animate changes from 1 -> 2 -> 3 dimensions

## Style
- [ ] Content is not correctly centered
- [ ] Remove the "waiting for review" thing (not publishing on Distill)
- [ ] Pick more attractive colors (the specific red and blue colors we have are yucky)
- [ ] initial zoom on the hero plot could be increased on some platforms. making it big is good.

## Polish
- [ ] Fix plot horizontal alignment with Distill column
- [ ] Improve 2D arrow aesthetics
- [x] Auto-orbit all 3D plots
  - Added controls.autoRotate = true to all 3D renders. Pauses on click-drag, resumes after 3s.
- [x] Larger 3D text labels
  - Doubled canvas size (512x128) and font size (48px bold) for Three.js sprite labels.
- [x] Default to 50K vocabulary
  - Changed default from small (10K) to medium (50K). 10K was missing too many words (superlatives, gendered pairs).


## Boo-boos?

- [ ] are some superlatives missing? (e.g., we have soft -> softer but appear to
  be missing softest).  Please check that all of the data from the notebook was
  carried over correctly.
