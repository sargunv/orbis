// Golden vectors for orb-symbol, computed by GL JS's own functions.
//
// Runs inside a GL JS checkout at the pinned commit; see README.md:
//   node orbis/generate.mjs <output dir>
//
// Inputs are seeded pseudo-random cases plus the cases of GL JS's own unit
// tests. Every number is written so that it parses back to the same double:
// finite numbers in shortest round-trip form, and -0, NaN and infinities as
// strings.

import fs from 'fs';
import path from 'path';
import Point from '@mapbox/point-geometry';
import {PbfReader} from 'pbf';
import {VectorTile} from '@mapbox/vector-tile';
import {Formatted} from '@maplibre/maplibre-gl-style-spec';

import {clipLine} from '../src/symbol/clip_line.ts';
import {checkMaxAngle} from '../src/symbol/check_max_angle.ts';
import {getAnchors, getCenterAnchor} from '../src/symbol/get_anchors.ts';
import {mergeLines} from '../src/symbol/merge_lines.ts';
import {Anchor} from '../src/symbol/anchor.ts';
import {getIconQuads} from '../src/symbol/quads.ts';
import {shapeIcon, fitIconToText, applyTextFit} from '../src/symbol/shaping.ts';
import {CollisionFeature} from '../src/symbol/collision_feature.ts';
import {evaluateVariableOffset, INVALID_TEXT_OFFSET} from '../src/style/style_layer/variable_text_anchor.ts';
import {findPoleOfInaccessibility} from '../src/util/find_pole_of_inaccessibility.ts';
import {CollisionBoxArray} from '../src/data/array_types.g.ts';
import {ImagePosition, ImageAtlas} from '../src/render/image_atlas.ts';
import {GlyphAtlas} from '../src/render/glyph_atlas.ts';
import {parseGlyphPbf} from '../src/style/parse_glyph_pbf.ts';
import {RGBAImage} from '../src/util/image.ts';
import {SymbolStyleLayer} from '../src/style/style_layer/symbol_style_layer.ts';
import {SymbolBucket} from '../src/data/bucket/symbol_bucket.ts';
import {EvaluationParameters} from '../src/style/evaluation_parameters.ts';
import {performSymbolLayout} from '../src/symbol/symbol_layout.ts';
import {transformText} from '../src/symbol/transform_text.ts';
import {getTextVariableAnchorOffset} from '../src/style/style_layer/variable_text_anchor.ts';
import {CanonicalTileID, OverscaledTileID} from '../src/tile/tile_id.ts';
import {FeatureIndex} from '../src/data/feature_index.ts';
import {SubdivisionGranularitySetting} from '../src/render/subdivision_granularity_settings.ts';
import {loadGeometry} from '../src/data/load_geometry.ts';

const [outDir] = process.argv.slice(2);
if (!outDir) throw new Error('usage: generate.mjs <out dir>');

function num(x: number): number | string {
    if (Number.isNaN(x)) return 'NaN';
    if (x === Infinity) return 'Infinity';
    if (x === -Infinity) return '-Infinity';
    if (Object.is(x, -0)) return '-0';
    return x;
}

function write(name: string, value: unknown) {
    const json = JSON.stringify(value, (_k, v) => typeof v === 'number' ? num(v) : v);
    fs.writeFileSync(path.join(outDir, name), `${json}\n`);
    console.log(`${name}: ${json.length} bytes`);
}

// mulberry32
function rng(seed: number) {
    return () => {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const random = rng(0x0b15);
const int = (lo: number, hi: number) => Math.floor(lo + random() * (hi - lo + 1));
const pick = <T>(xs: T[]): T => xs[Math.floor(random() * xs.length)];

const pts = (line: Point[]) => line.map(p => [p.x, p.y]);
const anchorOut = (a: Anchor) => [a.x, a.y, a.angle, a.segment === undefined ? null : a.segment];

// A wandering line of integer points, some of it outside the tile.
function randomLine(n: number, lo = -600, hi = 8800, step = 900): Point[] {
    const line = [new Point(int(lo, hi), int(lo, hi))];
    let angle = random() * Math.PI * 2;
    for (let i = 1; i < n; i++) {
        angle += (random() - 0.5) * pick([0.3, 1.2, 2.5]);
        const d = int(20, step);
        const prev = line[i - 1];
        line.push(new Point(Math.round(prev.x + Math.cos(angle) * d), Math.round(prev.y + Math.sin(angle) * d)));
    }
    return line;
}

// ---- leaves ----

const leaves: Record<string, unknown[]> = {};

leaves.clipLine = [];
{
    const cases: {lines: Point[][]; box: number[]}[] = [
        {lines: [[new Point(-10, -10), new Point(10, 10)]], box: [0, 0, 8192, 8192]},
        {lines: [[new Point(0, 0), new Point(8192, 8192)]], box: [0, 0, 8192, 8192]},
        {lines: [[new Point(-5, 5), new Point(5, 5), new Point(5, -5), new Point(5, 15), new Point(15, 15)]], box: [0, 0, 10, 10]},
        {lines: [[new Point(100, 100)]], box: [0, 0, 8192, 8192]},
    ];
    for (let i = 0; i < 60; i++) {
        const lines = [];
        for (let j = int(1, 3); j > 0; j--) lines.push(randomLine(int(2, 14)));
        cases.push({lines, box: pick([[0, 0, 8192, 8192], [-128, -128, 8320, 8320], [1000, 2000, 5000, 4000]])});
    }
    for (const c of cases) {
        const out = clipLine(c.lines, c.box[0], c.box[1], c.box[2], c.box[3]);
        leaves.clipLine.push({lines: c.lines.map(pts), box: c.box, out: out.map(pts)});
    }
}

leaves.checkMaxAngle = [];
{
    const cases: {line: Point[]; anchor: Anchor; labelLength: number; windowSize: number; maxAngle: number}[] = [];
    // GL JS check_max_angle.test.ts
    const line = [new Point(0, 0), new Point(0, 100), new Point(10, 120), new Point(0, 140), new Point(0, 240)];
    for (const [labelLength, windowSize, maxAngle] of [[30, 20, Math.PI], [30, 20, Math.PI / 4], [60, 10, Math.PI / 2], [400, 10, Math.PI]]) {
        cases.push({line, anchor: new Anchor(0, 120, 0, 2), labelLength, windowSize, maxAngle});
    }
    cases.push({line, anchor: new Anchor(0, 120, 0), labelLength: 30, windowSize: 20, maxAngle: 0.1});
    cases.push({line, anchor: new Anchor(0, 120, 0, 2), labelLength: 0, windowSize: 20, maxAngle: 0.1});
    for (let i = 0; i < 120; i++) {
        const l = randomLine(int(3, 16), 0, 8192, 300);
        const segment = int(0, l.length - 2);
        const t = random();
        const a = l[segment], b = l[segment + 1];
        const anchor = new Anchor(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, b.angleTo(a), segment);
        anchor._round();
        cases.push({line: l, anchor, labelLength: int(0, 1500), windowSize: int(5, 200), maxAngle: pick([0.2, 0.6, 38 / 180 * Math.PI, Math.PI / 4, Math.PI])});
    }
    for (const c of cases) {
        leaves.checkMaxAngle.push({
            line: pts(c.line), anchor: anchorOut(c.anchor), labelLength: c.labelLength,
            windowSize: c.windowSize, maxAngle: c.maxAngle,
            out: checkMaxAngle(c.line, c.anchor, c.labelLength, c.windowSize, c.maxAngle)
        });
    }
}

const box = (left: number, right: number) => ({left, right, top: -10, bottom: 10});

leaves.getAnchors = [];
leaves.getCenterAnchor = [];
{
    // GL JS get_anchors.test.ts
    const nonContinued = [new Point(1, 1), new Point(1, 3.1), new Point(3, 6), new Point(7, 6)];
    const continued = [new Point(0, 0), ...nonContinued];
    const cases: {line: Point[]; spacing: number; maxAngle: number; text: any; icon: any; glyphSize: number; boxScale: number; overscaling: number}[] = [];
    for (const l of [nonContinued, continued]) {
        cases.push({line: l, spacing: 2, maxAngle: Math.PI, text: box(-1, 1), icon: box(-0.5, 0.5), glyphSize: 1, boxScale: 1, overscaling: 1});
        cases.push({line: l, spacing: 2, maxAngle: Math.PI, text: box(-1, 1), icon: box(-0.5, 0.5), glyphSize: 1, boxScale: 1, overscaling: 2});
        cases.push({line: l, spacing: 2, maxAngle: Math.PI, text: box(-10, 10), icon: null, glyphSize: 1, boxScale: 1, overscaling: 1});
        cases.push({line: l, spacing: 2, maxAngle: Math.PI / 4, text: box(-0.5, 0.5), icon: box(-0.5, 0.5), glyphSize: 1, boxScale: 1, overscaling: 1});
    }
    for (let i = 0; i < 120; i++) {
        const l = randomLine(int(2, 18), -200, 8400, pick([60, 400, 1500]));
        const textWidth = random() < 0.8 ? int(10, 400) : null;
        const iconWidth = random() < 0.4 ? int(5, 60) : null;
        cases.push({
            line: l, spacing: pick([16, 250, 500, 4000]), maxAngle: pick([45, 38, 120, 180]) / 180 * Math.PI,
            text: textWidth === null ? null : box(-textWidth / 2, textWidth / 2),
            icon: iconWidth === null ? null : box(-iconWidth / 2, iconWidth / 2),
            glyphSize: 24, boxScale: pick([8, 16, 13.5, 8 / 3]), overscaling: pick([1, 1, 2, 4]),
        });
    }
    for (const c of cases) {
        const input = {line: pts(c.line), maxAngle: c.maxAngle, text: c.text, icon: c.icon, glyphSize: c.glyphSize, boxScale: c.boxScale};
        const anchors = getAnchors(c.line, c.spacing, c.maxAngle, c.text, c.icon, c.glyphSize, c.boxScale, c.overscaling, 8192);
        leaves.getAnchors.push({...input, spacing: c.spacing, overscaling: c.overscaling, out: anchors.map(anchorOut)});
        const center = getCenterAnchor(c.line, c.maxAngle, c.text, c.icon, c.glyphSize, c.boxScale);
        leaves.getCenterAnchor.push({...input, out: center ? anchorOut(center) : null});
    }
}

leaves.mergeLines = [];
{
    const lines1d = (spec: (string | number)[][]) => spec.map(([text, ...xs]) => ({text: text as string, line: (xs as number[]).map(x => new Point(x, 0))}));
    const cases = [
        lines1d([['a', 0, 1, 2], ['b', 4, 5, 6], ['a', 8, 9], ['a', 2, 3, 4], ['a', 6, 7, 8], ['a', 5, 6]]),
        lines1d([['a', 0, 1, 2], ['a', 4, 5, 6], ['a', 2, 3, 4]]),
        lines1d([['a', 0, 1, 2], ['a', 2, 3, 4], ['a', 4, 0]]),
        lines1d([['a', 4, 5], ['a', 3, 4], ['a', 2, 3], ['a', 1, 2], ['a', 0, 1]]),
        lines1d([['a', 0, 1], ['a', 2, 3], ['a', 4, 5], ['a', 1, 2], ['a', 3, 4]]),
    ];
    for (let i = 0; i < 80; i++) {
        const features = [];
        for (let j = int(1, 14); j > 0; j--) {
            const a = int(0, 8), b = int(0, 8);
            const mid = int(0, 8);
            features.push({text: pick(['a', 'b', '', null, 'a:1']), line: [new Point(a, a % 3), new Point(mid, 9), new Point(b, b % 3)]});
        }
        cases.push(features);
    }
    for (const features of cases) {
        const input = features.map((f, index) => ({index, text: f.text, line: pts(f.line)}));
        const symbolFeatures = features.map((f, index) => ({
            index, text: f.text === null ? undefined : Formatted.fromString(f.text), geometry: [f.line.map(p => p.clone())]
        }));
        const out = mergeLines(symbolFeatures as any);
        leaves.mergeLines.push({features: input, out: out.map((f: any) => ({index: f.index, line: pts(f.geometry[0])}))});
    }
}

leaves.pole = [];
{
    const square = [[new Point(0, 0), new Point(100, 0), new Point(100, 100), new Point(0, 100), new Point(0, 0)]];
    const withHole = [...square, [new Point(20, 20), new Point(20, 60), new Point(60, 60), new Point(60, 20), new Point(20, 20)]];
    const cases: Point[][][] = [square, withHole, [[new Point(5, 5), new Point(5, 5), new Point(5, 5)]]];
    for (let i = 0; i < 40; i++) {
        const cx = int(0, 8192), cy = int(0, 8192), n = int(3, 24);
        const outer: Point[] = [];
        for (let k = 0; k < n; k++) {
            const a = k / n * Math.PI * 2;
            const r = int(50, 2000) * (k % 2 && random() < 0.5 ? 0.3 : 1);
            outer.push(new Point(Math.round(cx + Math.cos(a) * r), Math.round(cy + Math.sin(a) * r)));
        }
        if (random() < 0.5) outer.push(outer[0].clone());
        const polygon = [outer];
        if (random() < 0.3) {
            const r = int(10, 40);
            polygon.push([new Point(cx - r, cy - r), new Point(cx - r, cy + r), new Point(cx + r, cy + r), new Point(cx + r, cy - r)]);
        }
        cases.push(polygon);
    }
    for (const polygon of cases) {
        for (const precision of [16, 1]) {
            const p = findPoleOfInaccessibility(polygon, precision);
            leaves.pole.push({polygon: polygon.map(pts), precision, out: [p.x, p.y]});
        }
    }
}

// Icons: images as orb-symbol's `IconImage` describes them.
type ImageSpec = {
    width: number; height: number; pixelRatio: number; sdf: boolean;
    stretchX?: [number, number][]; stretchY?: [number, number][];
    content?: [number, number, number, number];
    textFitWidth?: string; textFitHeight?: string;
};

function imagePosition(spec: ImageSpec, x = 0, y = 0): ImagePosition {
    return new ImagePosition({x, y, w: spec.width + 2, h: spec.height + 2}, {
        pixelRatio: spec.pixelRatio, version: 1, sdf: spec.sdf, stretchX: spec.stretchX, stretchY: spec.stretchY,
        content: spec.content, textFitWidth: spec.textFitWidth as any, textFitHeight: spec.textFitHeight as any,
    } as any);
}

function randomImage(): ImageSpec {
    const width = int(4, 60), height = int(4, 40);
    const spec: ImageSpec = {width, height, pixelRatio: pick([1, 2, 1.5]), sdf: random() < 0.3};
    if (random() < 0.6) {
        const zones = (size: number) => {
            const a = int(1, size / 2), b = int(a, size - 1);
            return random() < 0.4 ? [[a, b] as [number, number]] : [[1, a] as [number, number], [b, size - 1] as [number, number]];
        };
        if (random() < 0.8) spec.stretchX = zones(width);
        if (random() < 0.8) spec.stretchY = zones(height);
    }
    if (random() < 0.6) {
        spec.content = [int(0, 3), int(0, 3), width - int(0, 3), height - int(0, 3)];
        const fits = [undefined, 'stretchOrShrink', 'stretchOnly', 'proportional'];
        spec.textFitWidth = pick(fits);
        spec.textFitHeight = pick(fits);
    }
    return spec;
}

const anchors = ['center', 'left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right'];

const iconOut = (i: any) => ({top: i.top, bottom: i.bottom, left: i.left, right: i.right, collisionPadding: i.collisionPadding ?? null});

leaves.shapeIcon = [];
leaves.fitIconToText = [];
leaves.applyTextFit = [];
leaves.iconQuads = [];
leaves.collisionFeature = [];
{
    // GL JS shaping.test.ts applyTextFit cases.
    const textFitCases: [ImageSpec, number[]][] = [
        [{width: 98, height: 18, pixelRatio: 1, sdf: false, content: [5, 5, 95, 15]}, [0, 0, 14, 14]],
        [{width: 98, height: 18, pixelRatio: 1, sdf: false, content: [5, 5, 95, 15], textFitWidth: 'stretchOrShrink', textFitHeight: 'stretchOrShrink'}, [0, 0, 14, 14]],
        [{width: 98, height: 18, pixelRatio: 1, sdf: false, content: [5, 5, 95, 15], textFitWidth: 'stretchOnly', textFitHeight: 'proportional'}, [0, 0, 14, 14]],
        [{width: 18, height: 98, pixelRatio: 1, sdf: false, content: [5, 5, 15, 95], textFitWidth: 'proportional', textFitHeight: 'stretchOnly'}, [0, 0, 14, 14]],
    ];
    for (let i = 0; i < 60; i++) {
        const spec = randomImage();
        if (!spec.content) spec.content = [0, 0, spec.width, spec.height];
        const l = int(-80, 0), t = int(-40, 0);
        textFitCases.push([spec, [l, t, l + int(1, 160), t + int(1, 80)]]);
    }
    for (const [image, [left, top, right, bottom]] of textFitCases) {
        const icon = {image: imagePosition(image), left, top, right, bottom};
        const b = applyTextFit(icon as any);
        leaves.applyTextFit.push({image, icon: {left, top, right, bottom}, out: [b.x1, b.y1, b.x2, b.y2]});
    }

    for (let i = 0; i < 80; i++) {
        const image = randomImage();
        const offset = [int(-10, 10) / 2, int(-10, 10) / 2];
        const anchor = pick(anchors);
        const shaped = shapeIcon(imagePosition(image), offset as [number, number], anchor as any);
        leaves.shapeIcon.push({image, offset, anchor, out: iconOut(shaped)});

        const text = {left: int(-200, 0) + 0.25, right: int(0, 200) + 0.5, top: int(-60, 0) - 0.125, bottom: int(0, 60) + 0.75};
        const fit = pick(['width', 'height', 'both']);
        const padding = [int(0, 6), int(0, 6), int(0, 6), int(0, 6)];
        const fontScale = pick([16 / 24, 1, 12 / 24, 1.3]);
        const fitted = fitIconToText(shaped, text as any, fit, padding as any, offset as [number, number], fontScale);
        leaves.fitIconToText.push({image, icon: iconOut(shaped), text, fit, padding, offset, fontScale, out: iconOut(fitted)});

        if (i >= 30) continue;
        for (const [positioned, hasFit] of [[shaped, false], [fitted, true]] as const) {
            const rotate = pick([0, 0, 30, -45, 90]);
            const sdf = image.sdf;
            const quads = getIconQuads(positioned, rotate, sdf, hasFit);
            leaves.iconQuads.push({
                image, icon: iconOut(positioned), rotate, sdf, hasIconTextFit: hasFit,
                out: quads.map(q => ({
                    tl: [q.tl.x, q.tl.y], tr: [q.tr.x, q.tr.y], bl: [q.bl.x, q.bl.y], br: [q.br.x, q.br.y],
                    tex: [q.tex.x, q.tex.y, q.tex.w, q.tex.h],
                    pixelOffsetTL: [q.pixelOffsetTL.x, q.pixelOffsetTL.y], pixelOffsetBR: [q.pixelOffsetBR.x, q.pixelOffsetBR.y],
                    minFontScale: [q.minFontScaleX, q.minFontScaleY], sdf: q.isSDF,
                })),
            });

            {
                const alignLine = random() < 0.3;
                const boxScale = pick([8, 16, 2.5]);
                const pad = [int(0, 8), int(0, 8), int(0, 8), int(0, 8)];
                const r = pick([0, 0, 15, 90, -120]);
                const array = new CollisionBoxArray();
                const anchorPoint = new Anchor(int(0, 8191), int(0, 8191), 0);
                const feature = new CollisionFeature(array, anchorPoint, 0, 0, 0, positioned, boxScale, pad as any, alignLine, r);
                const boxOut = feature.boxEndIndex > feature.boxStartIndex ? (() => {
                    const b = array.get(feature.boxStartIndex);
                    return [b.x1, b.y1, b.x2, b.y2];
                })() : null;
                leaves.collisionFeature.push({
                    shape: iconOut(positioned), image, boxScale, padding: pad, alignLine, rotate: r,
                    out: {box: boxOut, circle: feature.circleDiameter ?? null},
                });
            }
        }
    }
    // Text shapes: no image.
    for (let i = 0; i < 40; i++) {
        const shape = {left: -int(0, 300) - 0.5, right: int(0, 300) + 0.25, top: -int(0, 40) - 0.75, bottom: int(-5, 40) + 0.5};
        for (const alignLine of [false, true]) {
            const boxScale = pick([8, 16, 13.5]);
            const pad = [2, 2, 2, 2].map(p => p * boxScale / 8);
            const r = pick([0, 0, 15, 90, -120]);
            const array = new CollisionBoxArray();
            const feature = new CollisionFeature(array, new Anchor(100, 100, 0), 0, 0, 0, shape, boxScale, pad as any, alignLine, r);
            const boxOut = feature.boxEndIndex > feature.boxStartIndex ? (() => {
                const b = array.get(feature.boxStartIndex);
                return [b.x1, b.y1, b.x2, b.y2];
            })() : null;
            leaves.collisionFeature.push({
                shape: {...shape, collisionPadding: null}, image: null, boxScale, padding: pad, alignLine, rotate: r,
                out: {box: boxOut, circle: feature.circleDiameter ?? null},
            });
        }
    }
}

leaves.variableOffset = [];
for (const anchor of anchors) {
    for (const offset of [[0, 0], [1.5 * 24, -2 * 24], [-12, 7.5], [24, INVALID_TEXT_OFFSET], [-5, INVALID_TEXT_OFFSET], [0, INVALID_TEXT_OFFSET]]) {
        leaves.variableOffset.push({anchor, offset, out: evaluateVariableOffset(anchor as any, offset as [number, number])});
    }
}

write('leaves.json', leaves);

// ---- whole layouts ----

type LayoutCase = {
    name: string;
    tile: string;
    tileId: [number, number, number];
    sourceLayer: string;
    zoom: number;
    overscaling?: number;
    pixelRatio?: number;
    promoteId?: string;
    layout: Record<string, unknown>;
    filter?: unknown;
    images?: Record<string, ImageSpec>;
};

const cases: LayoutCase[] = JSON.parse(fs.readFileSync(path.join(outDir, 'layout-cases.json'), 'utf8'));
const glyphRoot = path.join(outDir, 'glyphs');
const rangeCache = new Map<string, Map<number, any>>();

function pbfGlyph(stack: string, cp: number) {
    const range = Math.floor(cp / 256);
    const key = `${stack}/${range}`;
    if (!rangeCache.has(key)) {
        const file = path.join(glyphRoot, stack, `${range * 256}-${range * 256 + 255}.pbf`);
        const glyphs = new Map<number, any>();
        if (fs.existsSync(file)) for (const g of parseGlyphPbf(fs.readFileSync(file))) glyphs.set(g.id, g);
        rangeCache.set(key, glyphs);
    }
    return rangeCache.get(key).get(cp) ?? null;
}

function sectionsOut(text: Formatted) {
    return text.sections.map(s => ({
        text: s.text, image: s.image ? s.image.name : null, scale: s.scale ?? null,
        fontStack: s.fontStack ?? null, verticalAlign: s.verticalAlign ?? null,
    }));
}

const structOut = (s: any, fields: string[]) => Object.fromEntries(fields.map(f => [f, s[f]]));

for (const c of cases) {
    const vt = new VectorTile(new PbfReader(fs.readFileSync(path.join(outDir, 'tiles', c.tile))));
    const sourceLayer = vt.layers[c.sourceLayer];
    const [z, x, y] = c.tileId;
    const canonical = new CanonicalTileID(z, x, y);
    const layer = new SymbolStyleLayer({id: c.name, type: 'symbol', source: 's', 'source-layer': c.sourceLayer, layout: c.layout, filter: c.filter} as any, {});
    layer.recalculate(new EvaluationParameters(c.zoom), []);
    const collisionBoxArray = new CollisionBoxArray();
    const bucket = new SymbolBucket({
        zoom: c.zoom, overscaling: c.overscaling ?? 1, collisionBoxArray, layers: [layer], index: 0,
        pixelRatio: c.pixelRatio ?? 1, sourceLayerIndex: 0, sourceID: 's',
    } as any);
    const featureIndex = new FeatureIndex(new OverscaledTileID(z, 0, z, x, y), c.promoteId);
    const indexed = [];
    for (let i = 0; i < sourceLayer.length; i++) {
        const feature = sourceLayer.feature(i);
        const id = featureIndex.getId(feature, c.sourceLayer);
        indexed.push({feature, id, index: i, sourceLayerIndex: 0});
    }
    const options = {
        featureIndex, iconDependencies: {}, patternDependencies: {}, glyphDependencies: {}, dashDependencies: {},
        availableImages: Object.keys(c.images ?? {}), subdivisionGranularity: SubdivisionGranularitySetting.noSubdivision,
    };

    // The evaluated inputs orb-symbol gets, feature by feature, as populate
    // and performSymbolLayout evaluate them.
    const layout = layer.layout;
    const unevaluated = (layer as any)._unevaluatedLayout._values;
    const at = (name: string, zoom: number, feature: any) =>
        unevaluated[name].possiblyEvaluate(new EvaluationParameters(zoom), canonical).evaluate(feature, {}, canonical);
    // `SymbolBucket.populate`'s hasText and hasIcon.
    const textField = layout.get('text-field');
    const textFont = layout.get('text-font');
    const iconImage = layout.get('icon-image');
    const hasText =
        (textField.value.kind !== 'constant' ||
            (textField.value.value instanceof Formatted && !textField.value.value.isEmpty()) ||
            textField.value.value.toString().length > 0) &&
        (textFont.value.kind !== 'constant' || textFont.value.value.length > 0);
    const hasIcon = iconImage.value.kind !== 'constant' || !!iconImage.value.value || Object.keys(iconImage.parameters).length > 0;
    const features = [];
    const globals = new EvaluationParameters(c.zoom);
    for (const {feature, id, index} of indexed) {
        const evaluationFeature: any = {type: feature.type, id: feature.id, properties: feature.properties, geometry: loadGeometry(feature)};
        if (!(layer as any)._featureFilter.filter(globals, evaluationFeature, canonical)) continue;
        let text = null;
        if (hasText) {
            const resolved = layer.getValueAndResolveTokens('text-field', evaluationFeature, canonical, options.availableImages);
            text = transformText(Formatted.factory(resolved), layer, evaluationFeature);
        }
        let icon = null;
        if (hasIcon) {
            const resolved = layer.getValueAndResolveTokens('icon-image', evaluationFeature, canonical, options.availableImages);
            icon = typeof resolved === 'string' ? resolved : resolved?.name ?? null;
        }
        if (!text && !icon) continue;
        const ev = (name: string) => layout.get(name as any).evaluate(evaluationFeature, {}, canonical);
        const sizes = (name: string, data: any) => ({
            tile: ev(name), layout: at(name, c.zoom + 1, evaluationFeature), max: at(name, 18, evaluationFeature),
            stops: data.kind === 'composite' ? [at(name, data.minZoom, evaluationFeature), at(name, data.maxZoom, evaluationFeature)] : [0, 0],
        });
        const vao = layout.get('text-variable-anchor-offset')?.evaluate(evaluationFeature, {}, canonical);
        features.push({
            index, id: c.promoteId ? (id ?? null) : null,
            text: text ? sectionsOut(text) : null, icon,
            sortKey: bucket.sortFeaturesByKey ? layout.get('symbol-sort-key').evaluate(evaluationFeature, {}, canonical) : null,
            textFont: ev('text-font'),
            textSize: sizes('text-size', bucket.textSizeData), iconSize: sizes('icon-size', bucket.iconSizeData),
            textLetterSpacing: ev('text-letter-spacing'), textAnchor: ev('text-anchor'), textJustify: ev('text-justify'),
            textMaxWidth: ev('text-max-width'), textOffset: ev('text-offset'), textRadialOffset: ev('text-radial-offset'),
            textVariableAnchorOffset: vao ? vao.values : null,
            textRotate: ev('text-rotate'), iconOffset: ev('icon-offset'), iconAnchor: ev('icon-anchor'),
            iconRotate: ev('icon-rotate'), iconPadding: ev('icon-padding').values, heightOffset: ev('symbol-height-offset'),
        });
    }

    bucket.populate(indexed as any, options as any, canonical);

    // Glyphs as GL JS's GlyphManager resolves them for this style: one PBF
    // code point per request; clusters and vertical forms unavailable.
    const glyphMap: any = {};
    const glyphs = [];
    for (const stack in options.glyphDependencies) {
        glyphMap[stack] = {default: {}, vertical: {}};
        for (const variant of ['default', 'vertical']) {
            for (const grapheme in (options.glyphDependencies as any)[stack][variant] ?? {}) {
                const cps = [...grapheme];
                const glyph = variant === 'default' && cps.length === 1 ? pbfGlyph(stack, cps[0].codePointAt(0)) : null;
                glyphMap[stack][variant][grapheme] = glyph;
            }
        }
    }
    const atlas = new GlyphAtlas(glyphMap);
    for (const stack in glyphMap) {
        for (const variant in glyphMap[stack]) {
            for (const grapheme in glyphMap[stack][variant]) {
                const g = glyphMap[stack][variant][grapheme];
                const rect = atlas.positions[stack]?.[variant]?.[grapheme]?.rect ?? null;
                glyphs.push({stack, variant, grapheme, metrics: g ? g.metrics : null, bitmap: g ? [g.bitmap.width, g.bitmap.height] : null, rect: rect ? [rect.x, rect.y, rect.w, rect.h] : null});
            }
        }
    }

    const images: Record<string, any> = {};
    for (const [name, spec] of Object.entries(c.images ?? {})) {
        images[name] = {
            data: new RGBAImage({width: spec.width, height: spec.height}), pixelRatio: spec.pixelRatio, sdf: spec.sdf, version: 1,
            stretchX: spec.stretchX, stretchY: spec.stretchY, content: spec.content, textFitWidth: spec.textFitWidth, textFitHeight: spec.textFitHeight,
        };
    }
    const imageAtlas = new ImageAtlas(images, {});
    const imageRects = Object.fromEntries(Object.entries(imageAtlas.iconPositions).map(([n, p]) => [n, [p.paddedRect.x, p.paddedRect.y]]));

    performSymbolLayout({
        bucket, glyphMap, glyphPositions: atlas.positions, imageMap: images, imagePositions: imageAtlas.iconPositions,
        showCollisionBoxes: false, canonical, subdivisionGranularity: SubdivisionGranularitySetting.noSubdivision,
        hasPromoteId: c.promoteId != null,
    });

    const instances = [];
    for (let i = 0; i < bucket.symbolInstances.length; i++) {
        const s = bucket.symbolInstances.get(i);
        const boxAt = (start: number, end: number) => {
            if (end <= start) return null;
            const b = collisionBoxArray.get(start);
            return [b.x1, b.y1, b.x2, b.y2, b.anchorPointX, b.anchorPointY];
        };
        instances.push({
            ...structOut(s, ['anchorX', 'anchorY', 'rightJustifiedTextSymbolIndex', 'centerJustifiedTextSymbolIndex',
                'leftJustifiedTextSymbolIndex', 'verticalPlacedTextSymbolIndex', 'placedIconSymbolIndex',
                'verticalPlacedIconSymbolIndex', 'key', 'featureIndex', 'useRuntimeCollisionCircles', 'textBoxScale',
                'collisionCircleDiameter', 'textAnchorOffsetStartIndex', 'textAnchorOffsetEndIndex', 'heightOffset']),
            textBox: boxAt(s.textBoxStartIndex, s.textBoxEndIndex),
            verticalTextBox: boxAt(s.verticalTextBoxStartIndex, s.verticalTextBoxEndIndex),
            iconBox: boxAt(s.iconBoxStartIndex, s.iconBoxEndIndex),
            verticalIconBox: boxAt(s.verticalIconBoxStartIndex, s.verticalIconBoxEndIndex),
        });
    }
    const placedFields = ['anchorX', 'anchorY', 'glyphStartIndex', 'numGlyphs', 'lineStartIndex', 'lineLength', 'segment',
        'lowerSize', 'upperSize', 'lineOffsetX', 'lineOffsetY', 'writingMode', 'associatedIconIndex', 'heightOffset'];
    const stream = (buffers: any) => {
        const placed = [];
        for (let i = 0; i < buffers.placedSymbolArray.length; i++) placed.push(structOut(buffers.placedSymbolArray.get(i), placedFields));
        // a_pos_offset (4 × i16), a_data (4 × u16), a_pixeloffset (4 × i16), a_height_offset (f32).
        const array = buffers.layoutVertexArray;
        const view = new DataView(array.arrayBuffer);
        const vertices = [];
        for (let v = 0; v < array.length; v++) {
            const o = v * array.bytesPerElement;
            const i16 = (k: number) => view.getInt16(o + k * 2, true);
            const u16 = (k: number) => view.getUint16(o + k * 2, true);
            vertices.push([i16(0), i16(1), i16(2), i16(3), u16(4), u16(5), u16(6), u16(7), i16(8), i16(9), i16(10), i16(11), view.getFloat32(o + 24, true)]);
        }
        return {placed, vertices};
    };
    const glyphOffsets = [];
    for (let i = 0; i < bucket.glyphOffsetArray.length; i++) glyphOffsets.push(bucket.glyphOffsetArray.getoffsetX(i));
    const lines = [];
    for (let i = 0; i < bucket.lineVertexArray.length; i++) lines.push([bucket.lineVertexArray.getx(i), bucket.lineVertexArray.gety(i)]);
    const anchorOffsets = [];
    for (let i = 0; i < bucket.textAnchorOffsets.length; i++) {
        const a = bucket.textAnchorOffsets.get(i);
        anchorOffsets.push([a.textAnchor, a.textOffset0, a.textOffset1]);
    }

    write(`layout-${c.name}.json`, {
        case: c,
        layer: {
            textSize: bucket.textSizeData, iconSize: bucket.iconSizeData,
            writingModes: layout.get('text-writing-mode'), variableAnchor: layout.get('text-variable-anchor') ?? null,
            radialOffsetSet: unevaluated['text-radial-offset'].value !== undefined,
            iconRotateConstant: layout.get('icon-rotate').isConstant() ? layout.get('icon-rotate').constantOr(0) : null,
            textRotationAlignment: layout.get('text-rotation-alignment'),
            iconRotationAlignment: layout.get('icon-rotation-alignment').constantOr('auto'),
            placement: layout.get('symbol-placement'), spacing: layout.get('symbol-spacing'),
            keepUpright: layout.get('text-keep-upright'), maxAngle: layout.get('text-max-angle'),
            lineHeight: layout.get('text-line-height'), textPadding: layout.get('text-padding'),
            iconTextFit: layout.get('icon-text-fit'), iconTextFitPadding: layout.get('icon-text-fit-padding'),
        },
        features, glyphs, imageRects,
        out: {
            instances, text: stream(bucket.text), icon: stream(bucket.icon), glyphOffsets, lines, anchorOffsets,
            sortKeyRanges: bucket.sortKeyRanges, sdfIcons: bucket.sdfIcons ?? null, iconsInText: bucket.iconsInText,
            iconsNeedLinear: bucket.iconsNeedLinear, maxHeightOffset: bucket.maxHeightOffset,
        },
    });
}
