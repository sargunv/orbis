# Test fonts

Subsets of Noto Sans (Regular) from
[notofonts.github.io](https://github.com/notofonts/notofonts.github.io),
unhinted TTF, cut with fontTools' `pyftsubset` to the characters of
`orb-text`'s script matrix (`src/matrix.rs`) with all layout features; the
Arabic subset also keeps the presentation-form blocks Arabic shaping maps to.

Noto Sans is © Google LLC and the Noto Project Authors, under the
[SIL Open Font License 1.1](https://openfontlicense.org/).

`noto-sans-hebrew-latin-400-normal.woff2` is the file of
[`@fontsource/noto-sans-hebrew`](https://fontsource.org/fonts/noto-sans-hebrew)
5.3.0 that GL JS's `font-faces` render fixture declares, unchanged; the same
licence applies.
