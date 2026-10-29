/* Hand-written SVG charts. Every function is pure: data in, SVG/HTML string out (testable in Node).
 * Number formatting follows pt-BR (4,17 ; 1.115.394).
 */
(function (root) {
  'use strict';

  let uid = 0;

  function esc(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ------------------------------------------------------------------ pt-BR numbers
  function groupThousands(intText) {
    return intText.replace(/\B(?=(\d{3})+(?!\d))/g, '.');
  }

  function fmt(x, decimals) {
    if (x === null || x === undefined || Number.isNaN(Number(x))) return '–';
    const d = decimals === undefined ? 2 : decimals;
    const value = Number(x);
    if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
    const fixed = Math.abs(value).toFixed(d);
    const parts = fixed.split('.');
    const sign = value < 0 && Number(fixed) !== 0 ? '−' : '';
    return sign + groupThousands(parts[0]) + (parts[1] ? ',' + parts[1] : '');
  }

  function fmtInt(n) {
    return fmt(Math.round(Number(n)), 0);
  }

  function fmtShape(shape) {
    if (!shape) return '';
    return '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';
  }

  // ------------------------------------------------------------------ colors
  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function mix(c1, c2, t) {
    const a = hexToRgb(c1);
    const b = hexToRgb(c2);
    const c = a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); });
    return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
  }

  const PALETTES = {
    diverging: ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'],
    blues: ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
    heat: ['#fff7ec', '#fdd49e', '#fc8d59', '#d7301f', '#7f0000'],
  };

  function rampColor(t, palette) {
    const stops = PALETTES[palette] || PALETTES.blues;
    const x = Math.max(0, Math.min(1, Number(t) || 0)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    return mix(stops[i], stops[i + 1], x - i);
  }

  function divergingColor(value, maxAbs) {
    if (!(maxAbs > 0)) return rampColor(0.5, 'diverging');
    const t = Math.max(-1, Math.min(1, value / maxAbs));
    return rampColor((t + 1) / 2, 'diverging');
  }

  function sequentialColor(value, min, max, palette) {
    const span = max - min;
    return rampColor(span > 0 ? (value - min) / span : 0, palette);
  }

  function inkFor(rgb) {
    const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(rgb);
    if (!m) return '#1d2433';
    const lum = 0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3];
    return lum < 140 ? '#ffffff' : '#1d2433';
  }

  function tokenColor(id) {
    const hue = (Number(id) * 137.508) % 360;
    return 'hsl(' + hue.toFixed(1) + ', 70%, 87%)';
  }

  // ------------------------------------------------------------------ axes helpers
  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const nice = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
    return nice * mag;
  }

  function niceTicks(min, max, count) {
    if (!(max > min)) { max = min + 1; }
    const step = niceStep(max - min, count || 5);
    const start = Math.ceil(min / step - 1e-9) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
    return { ticks: ticks, step: step };
  }

  function decimalsFor(step) {
    if (step >= 1) return 0;
    return Math.min(4, Math.ceil(-Math.log10(step)));
  }

  function svgOpen(width, height, cls, label) {
    return '<svg class="chart ' + (cls || '') + '" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + width + ' ' + height +
      '" width="' + width + '" height="' + height + '" role="img"' + (label ? ' aria-label="' + esc(label) + '"' : '') + '>';
  }

  // ------------------------------------------------------------------ horizontal bars
  /* items: [{label, value, title?, highlight?}] ; values may be negative (logits). */
  function barsH(items, options) {
    const o = Object.assign({ width: 560, barHeight: 18, gap: 5, labelWidth: 48, valueWidth: 72, decimals: 3,
      color: '#0f766e', negColor: '#4c6ef5', highlightColor: '#e8590c', label: '', maxValue: null }, options || {});
    const n = items.length;
    const height = Math.max(1, n) * (o.barHeight + o.gap) + o.gap + 4;
    const values = items.map(function (d) { return Number(d.value) || 0; });
    const lo = Math.min(0, Math.min.apply(null, values.concat([0])));
    let hi = Math.max(0, Math.max.apply(null, values.concat([0])));
    if (o.maxValue !== null && o.maxValue > hi) hi = o.maxValue;
    const plotX = o.labelWidth + 6;
    const plotW = o.width - plotX - o.valueWidth - 6;
    const span = hi - lo || 1;
    const x = function (v) { return plotX + (v - lo) / span * plotW; };
    const zero = x(0);
    let svg = svgOpen(o.width, height, 'bars', o.label);
    if (lo < 0) {
      svg += '<line x1="' + zero.toFixed(1) + '" y1="0" x2="' + zero.toFixed(1) + '" y2="' + height + '" class="axis-zero"/>';
    }
    items.forEach(function (d, i) {
      const v = values[i];
      const y = o.gap + i * (o.barHeight + o.gap);
      const x0 = Math.min(zero, x(v));
      const w = Math.max(v === 0 ? 0 : 1, Math.abs(x(v) - zero));
      const fill = d.highlight ? o.highlightColor : (v < 0 ? o.negColor : o.color);
      const title = d.title || (d.label + ': ' + fmt(v, o.decimals));
      svg += '<g class="bar"><title>' + esc(title) + '</title>' +
        '<text x="' + (o.labelWidth) + '" y="' + (y + o.barHeight * 0.72).toFixed(1) + '" class="bar-label" text-anchor="end">' + esc(d.label) + '</text>' +
        '<rect x="' + x0.toFixed(1) + '" y="' + y + '" width="' + w.toFixed(1) + '" height="' + o.barHeight + '" rx="2" fill="' + fill + '"/>' +
        '<text x="' + (o.width - 4) + '" y="' + (y + o.barHeight * 0.72).toFixed(1) + '" class="bar-value" text-anchor="end">' + esc(fmt(v, o.decimals)) + '</text>' +
        '</g>';
    });
    return svg + '</svg>';
  }

  // ------------------------------------------------------------------ line chart
  /* series: [{name, color, points: [[x, y], ...], faint?}] ; options.refLine: {y, label}.
   * Faint series (previous runs) are drawn thin, behind, without markers. The legend wraps. */
  function lineChart(series, options) {
    const o = Object.assign({ width: 640, height: 300, margin: { l: 56, r: 18, t: 16, b: 58 }, xLabel: '', yLabel: '',
      decimals: 2, refLine: null, xMin: 0, xMax: null, yMin: null, yMax: null, label: '' }, options || {});
    const m = o.margin;
    const pw = o.width - m.l - m.r;
    const ph = o.height - m.t - m.b;
    // legend layout first: it may need several rows
    const items = series.filter(function (s) { return !s.faint; }).concat(series.filter(function (s) { return s.faint; }))
      .map(function (s) { return { text: s.name, color: s.color, faint: !!s.faint }; });
    if (o.refLine) items.push({ text: o.refLine.label + ' = ' + fmt(o.refLine.y, o.decimals), ref: true });
    const rows = [[]];
    let rowW = 0;
    items.forEach(function (it) {
      const w = 36 + it.text.length * 6.9;
      if (rowW + w > o.width - m.l - 8 && rows[rows.length - 1].length) { rows.push([]); rowW = 0; }
      rows[rows.length - 1].push(Object.assign(it, { x: m.l + rowW }));
      rowW += w;
    });
    const height = o.height + (rows.length - 1) * 17;
    const all = [];
    series.forEach(function (s) { (s.points || []).forEach(function (p) { all.push(p); }); });
    let xMax = o.xMax !== null ? o.xMax : Math.max.apply(null, all.map(function (p) { return p[0]; }).concat([1]));
    const xMin = o.xMin;
    if (!(xMax > xMin)) xMax = xMin + 1;
    const ys = all.map(function (p) { return p[1]; });
    if (o.refLine) ys.push(o.refLine.y);
    let yMin = o.yMin !== null ? o.yMin : Math.min.apply(null, ys.concat([Infinity]));
    let yMax = o.yMax !== null ? o.yMax : Math.max.apply(null, ys.concat([-Infinity]));
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) { yMin = 0; yMax = 1; }
    if (yMax - yMin < 1e-6) { yMin -= 0.5; yMax += 0.5; }
    const pad = (yMax - yMin) * 0.08;
    yMin -= pad; yMax += pad;
    const yt = niceTicks(yMin, yMax, 5);
    const xt = niceTicks(xMin, xMax, 6);
    const sx = function (v) { return m.l + (v - xMin) / (xMax - xMin) * pw; };
    const sy = function (v) { return m.t + (1 - (v - yMin) / (yMax - yMin)) * ph; };
    const yDec = Math.max(1, decimalsFor(yt.step));
    let svg = svgOpen(o.width, height, 'line', o.label);
    yt.ticks.forEach(function (v) {
      if (v < yMin || v > yMax) return;
      svg += '<line x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + sy(v).toFixed(1) + '" y2="' + sy(v).toFixed(1) + '" class="grid"/>' +
        '<text x="' + (m.l - 6) + '" y="' + (sy(v) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(fmt(v, yDec)) + '</text>';
    });
    xt.ticks.forEach(function (v) {
      if (v < xMin || v > xMax) return;
      svg += '<line x1="' + sx(v).toFixed(1) + '" x2="' + sx(v).toFixed(1) + '" y1="' + (m.t + ph) + '" y2="' + (m.t + ph + 4) + '" class="axis"/>' +
        '<text x="' + sx(v).toFixed(1) + '" y="' + (m.t + ph + 17) + '" class="tick" text-anchor="middle">' + esc(fmtInt(v)) + '</text>';
    });
    svg += '<line x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + (m.t + ph) + '" y2="' + (m.t + ph) + '" class="axis"/>' +
      '<line x1="' + m.l + '" x2="' + m.l + '" y1="' + m.t + '" y2="' + (m.t + ph) + '" class="axis"/>';
    if (o.xLabel) svg += '<text x="' + (m.l + pw / 2) + '" y="' + (m.t + ph + 34) + '" class="axis-label" text-anchor="middle">' + esc(o.xLabel) + '</text>';
    if (o.yLabel) svg += '<text transform="translate(14,' + (m.t + ph / 2) + ') rotate(-90)" class="axis-label" text-anchor="middle">' + esc(o.yLabel) + '</text>';
    if (o.refLine && Number.isFinite(o.refLine.y)) {
      const y = sy(o.refLine.y).toFixed(1);
      svg += '<g class="ref"><title>' + esc(o.refLine.label + ': ' + fmt(o.refLine.y, o.decimals)) + '</title>' +
        '<line x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + y + '" y2="' + y + '" class="ref-line"/></g>';
    }
    const ordered = series.filter(function (s) { return s.faint; }).concat(series.filter(function (s) { return !s.faint; }));
    ordered.forEach(function (s) {
      const pts = (s.points || []).filter(function (p) { return Number.isFinite(p[1]); });
      if (!pts.length) return;
      const d = pts.map(function (p, i) { return (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1); }).join(' ');
      if (s.faint) {
        svg += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="1.6" stroke-opacity="0.85" stroke-linejoin="round">' +
          '<title>' + esc(s.name) + '</title></path>';
        return;
      }
      svg += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="2.2" stroke-linejoin="round"/>';
      pts.forEach(function (p) {
        svg += '<circle cx="' + sx(p[0]).toFixed(1) + '" cy="' + sy(p[1]).toFixed(1) + '" r="3.2" fill="' + s.color + '">' +
          '<title>' + esc(s.name + ' · passo ' + fmtInt(p[0]) + ': ' + fmt(p[1], 3)) + '</title></circle>';
      });
    });
    rows.forEach(function (row, r) {
      const ly = o.height - 10 + r * 17;
      row.forEach(function (it) {
        const line = it.ref ? ' class="ref-line"' : ' stroke="' + it.color + '" stroke-width="' + (it.faint ? 1.6 : 2.5) + '"';
        svg += '<line x1="' + it.x + '" x2="' + (it.x + 22) + '" y1="' + (ly - 4) + '" y2="' + (ly - 4) + '"' + line + '/>' +
          '<text x="' + (it.x + 28) + '" y="' + ly + '" class="legend">' + esc(it.text) + '</text>';
      });
    });
    return svg + '</svg>';
  }

  // ------------------------------------------------------------------ heatmap
  /* matrix: rows of numbers. options.scale: 'diverging' (signed, centred on 0) | 'blues' | 'heat' */
  function heatmap(matrix, options) {
    const rows = matrix.length;
    const cols = rows ? matrix[0].length : 0;
    const o = Object.assign({ width: 640, maxCell: 26, minCell: 3, cellAspect: 1, rowLabels: null, colLabels: null,
      scale: 'diverging', decimals: 3, rowTitle: 'row', colTitle: 'column', legendLabel: 'value', label: '',
      labelEvery: 1, min: null, max: null, maxHeight: null }, options || {});
    let min = Infinity;
    let max = -Infinity;
    matrix.forEach(function (row) { row.forEach(function (v) { if (v < min) min = v; if (v > max) max = v; }); });
    if (!Number.isFinite(min)) { min = 0; max = 0; }
    if (o.min !== null) min = o.min;
    if (o.max !== null) max = o.max;
    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const rowLabelW = o.rowLabels ? 30 : 6;
    const colLabelH = o.colLabels ? 22 : 6;
    let cell = Math.floor((o.width - rowLabelW - 8) / Math.max(1, cols));
    cell = Math.max(o.minCell, Math.min(o.maxCell, cell));
    let cellH = Math.max(o.minCell, Math.round(cell * o.cellAspect));
    if (o.maxHeight && rows * cellH > o.maxHeight) cellH = Math.max(o.minCell, Math.floor(o.maxHeight / rows));
    const gridW = cols * cell;
    const gridH = rows * cellH;
    const legendH = 46;
    const legendW = Math.min(260, Math.max(140, gridW));
    const legendTextW = (o.legendLabel.length + (o.scale === 'diverging' ? 25 : 0)) * 6.4;
    const width = Math.max(rowLabelW + gridW + 8, rowLabelW + legendW + 10 + legendTextW);
    const height = colLabelH + gridH + legendH;
    const color = function (v) {
      return o.scale === 'diverging' ? divergingColor(v, maxAbs) : sequentialColor(v, min, max, o.scale);
    };
    const labelFont = Math.max(7, Math.min(13, Math.min(cell, cellH) - 1));
    let svg = svgOpen(width, height, 'heatmap', o.label);
    if (o.colLabels) {
      o.colLabels.forEach(function (lab, c) {
        if (c % o.labelEvery) return;
        svg += '<text x="' + (rowLabelW + c * cell + cell / 2).toFixed(1) + '" y="' + (colLabelH - 6) + '" class="hm-label" text-anchor="middle" font-size="' + labelFont + '">' + esc(lab) + '</text>';
      });
    }
    for (let r = 0; r < rows; r++) {
      const y = colLabelH + r * cellH;
      if (o.rowLabels && !(r % o.labelEvery)) {
        svg += '<text x="' + (rowLabelW - 5) + '" y="' + (y + cellH * 0.78).toFixed(1) + '" class="hm-label" text-anchor="end" font-size="' + Math.max(7, Math.min(13, cellH - 1)) + '">' + esc(o.rowLabels[r]) + '</text>';
      }
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c];
        const rl = o.rowLabels ? o.rowLabels[r] : r;
        const cl = o.colLabels ? o.colLabels[c] : c;
        svg += '<rect x="' + (rowLabelW + c * cell) + '" y="' + y + '" width="' + cell + '" height="' + cellH + '" fill="' + color(v) + '">' +
          '<title>' + esc(o.rowTitle + ' ' + rl + ' · ' + o.colTitle + ' ' + cl + ': ' + fmt(v, o.decimals)) + '</title></rect>';
      }
    }
    // color legend
    const gid = 'hmg' + (++uid);
    const lx = rowLabelW;
    const lw = legendW;
    const ly = colLabelH + gridH + 12;
    const stops = o.scale === 'diverging' ? PALETTES.diverging : (PALETTES[o.scale] || PALETTES.blues);
    svg += '<defs><linearGradient id="' + gid + '" x1="0" x2="1" y1="0" y2="0">' +
      stops.map(function (c, i) { return '<stop offset="' + (i / (stops.length - 1)).toFixed(3) + '" stop-color="' + c + '"/>'; }).join('') +
      '</linearGradient></defs>' +
      '<rect x="' + lx + '" y="' + ly + '" width="' + lw + '" height="10" fill="url(#' + gid + ')" stroke="#c5ccd6" stroke-width="0.5"/>';
    const lo = o.scale === 'diverging' ? -maxAbs : min;
    const hiV = o.scale === 'diverging' ? maxAbs : max;
    svg += '<text x="' + lx + '" y="' + (ly + 24) + '" class="tick">' + esc(fmt(lo, o.decimals)) + '</text>' +
      '<text x="' + (lx + lw) + '" y="' + (ly + 24) + '" class="tick" text-anchor="end">' + esc(fmt(hiV, o.decimals)) + '</text>' +
      '<text x="' + (lx + lw / 2) + '" y="' + (ly + 24) + '" class="tick" text-anchor="middle">' + esc(o.scale === 'diverging' ? '0' : o.legendLabel) + '</text>' +
      '<text x="' + (lx + lw + 10) + '" y="' + (ly + 9) + '" class="legend">' + esc(o.legendLabel) + (o.scale === 'diverging' ? ' (blue < 0 < red)' : '') + '</text>';
    return svg + '</svg>';
  }

  // ------------------------------------------------------------------ scatter with text labels
  /* points: [{x, y, label, color?, title?}] ; options.legend: [{label, color}] */
  function scatter(points, options) {
    const o = Object.assign({ width: 640, height: 420, margin: { l: 44, r: 16, t: 14, b: 54 }, xLabel: '', yLabel: '',
      legend: null, label: '', fontSize: 13 }, options || {});
    const m = o.margin;
    const pw = o.width - m.l - m.r;
    const ph = o.height - m.t - m.b;
    const xs = points.map(function (p) { return p.x; });
    const ys = points.map(function (p) { return p.y; });
    let x0 = Math.min.apply(null, xs.concat([0]));
    let x1 = Math.max.apply(null, xs.concat([0]));
    let y0 = Math.min.apply(null, ys.concat([0]));
    let y1 = Math.max.apply(null, ys.concat([0]));
    const px = (x1 - x0 || 1) * 0.07;
    const py = (y1 - y0 || 1) * 0.07;
    x0 -= px; x1 += px; y0 -= py; y1 += py;
    const sx = function (v) { return m.l + (v - x0) / (x1 - x0) * pw; };
    const sy = function (v) { return m.t + (1 - (v - y0) / (y1 - y0)) * ph; };
    let svg = svgOpen(o.width, o.height, 'scatter', o.label);
    svg += '<rect x="' + m.l + '" y="' + m.t + '" width="' + pw + '" height="' + ph + '" class="plot-bg"/>';
    svg += '<line x1="' + sx(0).toFixed(1) + '" x2="' + sx(0).toFixed(1) + '" y1="' + m.t + '" y2="' + (m.t + ph) + '" class="axis-zero"/>' +
      '<line x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + sy(0).toFixed(1) + '" y2="' + sy(0).toFixed(1) + '" class="axis-zero"/>';
    points.forEach(function (p) {
      const cx = sx(p.x).toFixed(1);
      const cy = sy(p.y).toFixed(1);
      const color = p.color || '#0f766e';
      svg += '<g class="pt"><title>' + esc(p.title || p.label) + '</title>' +
        '<circle cx="' + cx + '" cy="' + cy + '" r="3.6" fill="' + color + '" fill-opacity="0.85"/>' +
        '<text x="' + (Number(cx) + 5) + '" y="' + (Number(cy) - 4) + '" class="pt-label" font-size="' + o.fontSize + '" fill="' + color + '">' + esc(p.label) + '</text></g>';
    });
    if (o.xLabel) svg += '<text x="' + (m.l + pw / 2) + '" y="' + (m.t + ph + 18) + '" class="axis-label" text-anchor="middle">' + esc(o.xLabel) + '</text>';
    if (o.yLabel) svg += '<text transform="translate(14,' + (m.t + ph / 2) + ') rotate(-90)" class="axis-label" text-anchor="middle">' + esc(o.yLabel) + '</text>';
    if (o.legend) {
      let lx = m.l;
      const ly = o.height - 10;
      o.legend.forEach(function (item) {
        svg += '<circle cx="' + (lx + 5) + '" cy="' + (ly - 4) + '" r="4.5" fill="' + item.color + '"/>' +
          '<text x="' + (lx + 14) + '" y="' + ly + '" class="legend">' + esc(item.label) + '</text>';
        lx += 26 + item.label.length * 7.4;
      });
    }
    return svg + '</svg>';
  }

  // ------------------------------------------------------------------ token chips (HTML)
  /* tokens: [{id, show, title?}] */
  function tokenChips(tokens, options) {
    const o = Object.assign({ showId: true, showChar: true, cls: '' }, options || {});
    return '<div class="chips ' + esc(o.cls) + '">' + tokens.map(function (t) {
      const title = t.title || ("'" + t.show + "' → id " + t.id);
      return '<span class="chip" style="background:' + tokenColor(t.id) + '" title="' + esc(title) + '">' +
        (o.showChar ? '<span class="chip-ch">' + esc(t.show) + '</span>' : '') +
        (o.showId ? '<span class="chip-id">' + esc(String(t.id)) + '</span>' : '') + '</span>';
    }).join('') + '</div>';
  }

  const api = {
    esc: esc, fmt: fmt, fmtInt: fmtInt, fmtShape: fmtShape, mix: mix, rampColor: rampColor,
    divergingColor: divergingColor, sequentialColor: sequentialColor, inkFor: inkFor, tokenColor: tokenColor,
    niceTicks: niceTicks, barsH: barsH, lineChart: lineChart, heatmap: heatmap, scatter: scatter, tokenChips: tokenChips,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Charts = api;
})(typeof window !== 'undefined' ? window : this);
