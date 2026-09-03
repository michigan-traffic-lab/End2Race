# Figure Assets

The final assets for each numbered paper figure are stored in the corresponding
`fig1` through `fig5` directories. Figures 3 and 5 retain their source render
videos and selected final frames; decoded full-frame sequences are temporary.

## External asset provenance

`fig2/f1tenth-platform.png` is the F1TENTH NX vehicle photograph from the official
[F1TENTH build documentation](https://f1tenth.readthedocs.io/en/ros1/index.html).
The documentation is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Figures 3 and 5 regeneration

Run in the `end2race` conda environment on the Austin map. Render at
`1728x972`, use a fixed `1.5x` camera, then crop `(378, 0, 1350, 972)` to obtain
the final `972x972` panels. Vehicle colors are ego RGB `(40,110,220)` and
opponent RGB `(225,60,60)`. Annotations use Nimbus Roman Bold: timestamps at
`(48,48)`, 76 px, RGB `(31,35,38)`; speeds at y=`788` and `872`, 72 px, in the
corresponding vehicle color. Speeds use one decimal place and frame labels use
`(I)`, `(II)`, `(III)`. Borders are defined in `report/main.tex` as 0.2 pt
`black!10`. Figure 3 draws the planned trajectory as a gray line without
lattice dots.

### Figure 3

Generate with `expert/collect.py`: ego index `1205`, interval `15`, opponent
`raceline0`, speed scale `0.8`, duration `8.0`. The source stem is
`o_ol0_e1205_i15_o1207_s0.8`. Use FFmpeg zero-based frames `192,324,372`, with
frame `324` as the camera reference and final world center
`(29.886022259673,1.659238131821)`. Times are `1.6,2.7,3.1` s; ego speeds are
`7.3,7.3,6.5` m/s; opponent speeds are `6.0,5.9,5.7` m/s. Center both speed
lines at x=`486`; use a `3.5` px line width for the track and gray trajectory.

### Figure 5

Generate with `evaluation/eval_multi.py`, `checkpoint/ppo.pt` (SHA-256
`5326715e83f863770525d6b6a9748e3981f58b225163167fc49f78cf228c09f7`),
ego `raceline1`, interval `15`, opponent speed scale `0.8`, duration `8.0`, no
noise, and seed `42`. Frame numbers below are one-based decoded-image numbers.

- Overtake 1 (`overtake-1_o_ol1_e0_o15_s0.8.mp4`): ego `0`, opponent `raceline1`; frames `280,350,420`, reference
  `350`, center `(120.797601779888,35.538250256554)`; times `2.3,2.9,3.5` s;
  ego speeds `9.1,9.1,9.5`, opponent `6.0,6.0,6.0` m/s; speed text centered at
  x=`620`.
- Overtake 2 (`overtake-2_o_ol1_e1362_o1377_s0.8.mp4`): ego `1362`, opponent `raceline1`; frames `200,310,390`, reference
  `310`, center `(33.196826157080,24.586532336468)`; times `1.7,2.6,3.2` s;
  ego speeds `8.3,5.6,7.6`, opponent `4.7,3.3,3.9` m/s; speed text left-aligned
  at x=`82`.
- Following (`follow_f_ol2_e550_o568_s0.8.mp4`): ego `550`, opponent `raceline2`; frames `650,750,900`, reference
  `750`, center `(38.210172751372,-27.102098476379)`; times `5.4,6.3,7.5` s;
  ego speeds `4.7,0.9,3.0`, opponent `3.9,3.7,6.0` m/s; speed text left-aligned
  at x=`74`.
- Collision (`collision_c_ol1_e236_o251_s0.8.mp4`): ego `236`, opponent `raceline1`; frames `301,349,385`, reference
  `350`, center `(81.428161794,20.912728767)`; times `2.5,2.9,3.2` s; ego
  speeds `7.2,5.7,5.5`, opponent `5.1,4.8,4.9` m/s; speed text centered at
  x=`486`.

Apply the listed camera and color overrides only in the temporary rendering
process; the repository renderer remains at its defaults. Retain the source
video and named final PNG panels in `fig3` or `fig5`, and discard only decoded
full-frame sequences.
