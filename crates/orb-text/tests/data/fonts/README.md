# Test fonts

Subsets of Noto Sans (Regular) from
[notofonts.github.io](https://github.com/notofonts/notofonts.github.io),
unhinted TTF, cut with fontTools' `pyftsubset` to the characters of
`orb-text`'s script tests (`src/tests.rs`) with all layout features; the
Arabic subset also keeps the presentation-form blocks Arabic shaping maps to.

Noto Sans is © Google LLC and the Noto Project Authors, under the
[SIL Open Font License 1.1](https://openfontlicense.org/).

`noto-sans-hebrew-latin-400-normal.woff2` is the file of
[`@fontsource/noto-sans-hebrew`](https://fontsource.org/fonts/noto-sans-hebrew)
5.3.0 that GL JS's `font-faces` render fixture declares, unchanged; the same
licence applies.

`NotoSansCJKjp-VF.otf` is Noto Sans CJK JP's variable font (2.004), whose
default instance is Thin, cut to "中A" with all layout features
(`pyftsubset NotoSansCJK-VF.otf.ttc --font-number=0 --text=中A
--layout-features='*' --name-IDs='*'`); `NotoSansCJKjp-Regular.otf` and
`NotoSansCJKjp-Thin.otf` are its static instances at weights 400 and 100
(`fonttools varLib.instancer NotoSansCJKjp-VF.otf wght=400`). Noto Sans CJK
is © Adobe, under the same licence.
