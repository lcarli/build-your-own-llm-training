/* Markdown subset renderer for the lab guides.
 * Supports: # ## ### headings, paragraphs, **bold**, *italic*, `code`, fenced code blocks,
 * - and 1. lists, > blockquotes (rendered as a "Tip" callout), | tables |, --- rules, [links](url),
 * and the workshop directives @code <brick>, @verify and @answer <inline markdown>.
 * Raw HTML is never passed through: every piece of text is escaped.
 */
(function (root) {
  'use strict';

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ------------------------------------------------------------------ inline
  function renderEmphasis(html) {
    return html
      .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*\w])\*(?=[^\s*])([^*]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/(^|[^*\w])\*([^\s*])\*(?!\*)/g, '$1<em>$2</em>');
  }

  function renderInline(text) {
    const slots = [];
    const keep = function (html) {
      slots.push(html);
      return '\u0000' + (slots.length - 1) + '\u0000';
    };
    let s = String(text).replace(/\u0000/g, '');
    s = s.replace(/`([^`]+)`/g, function (_m, code) {
      return keep('<code>' + escapeHtml(code) + '</code>');
    });
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_m, label, url) {
      const inner = renderEmphasis(escapeHtml(label));
      if (!/^(https?:\/\/|mailto:|#)/i.test(url)) return keep(inner);
      return keep('<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">' + inner + '</a>');
    });
    s = renderEmphasis(escapeHtml(s));
    return s.replace(/\u0000(\d+)\u0000/g, function (_m, i) { return slots[Number(i)]; });
  }

  // ------------------------------------------------------------------ Python highlighting
  const PY_KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else ' +
    'except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '));
  const PY_BUILTINS = new Set(('print len range int float str list dict set tuple sorted enumerate zip super ' +
    'isinstance sum min max abs round type object open').split(' '));
  const PY_TOKEN = /(#[^\n]*)|((?:[rRbBuUfF]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[A-Za-z_][\w.]*)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

  function span(cls, text) {
    return '<span class="tok-' + cls + '">' + escapeHtml(text) + '</span>';
  }

  function highlightPython(code) {
    let out = '';
    let last = 0;
    let expectName = null;
    let m;
    PY_TOKEN.lastIndex = 0;
    while ((m = PY_TOKEN.exec(code)) !== null) {
      out += escapeHtml(code.slice(last, m.index));
      const tok = m[0];
      if (m[1]) out += span('com', tok);
      else if (m[2]) out += span('str', tok);
      else if (m[3]) out += span('dec', tok);
      else if (m[4]) out += span('num', tok);
      else if (expectName) { out += span(expectName, tok); expectName = null; }
      else if (PY_KEYWORDS.has(tok)) {
        out += span('kw', tok);
        if (tok === 'def') expectName = 'fn';
        else if (tok === 'class') expectName = 'cls';
      } else if (PY_BUILTINS.has(tok)) out += span('bi', tok);
      else if (tok === 'self') out += span('self', tok);
      else out += escapeHtml(tok);
      last = PY_TOKEN.lastIndex;
    }
    return out + escapeHtml(code.slice(last));
  }

  // ------------------------------------------------------------------ directives
  function renderCodeCard(brick, opts) {
    const info = (opts.codes || {})[brick];
    if (!info) {
      return '<div class="code-card code-card-missing">Code <code>' + escapeHtml(brick) + '</code> not found.</div>';
    }
    return '<div class="code-card" data-brick="' + escapeHtml(brick) + '">' +
      '<div class="code-card-head">' +
        '<div class="code-card-file"><span class="code-path">' + escapeHtml(info.path) + '</span>' +
        '<span class="code-instr">Paste below the line <code># ---- PASTE BELOW ----</code></span></div>' +
        '<div class="code-card-actions">' +
          '<button type="button" class="btn btn-small btn-copy" data-action="copy">Copy</button>' +
          '<button type="button" class="btn btn-small btn-ghost" data-action="write" aria-pressed="false">I want to write it</button>' +
        '</div>' +
      '</div>' +
      '<pre class="code"><code>' + highlightPython(info.code.replace(/\n$/, '')) + '</code></pre>' +
      '<div class="code-hidden" hidden><p>Code hidden: write it yourself using the explanation, then click Verify.</p>' +
        '<button type="button" class="btn btn-small" data-action="show">Show code</button></div>' +
    '</div>';
  }

  function renderVerify(opts) {
    return '<div class="verify-box" data-lab="' + escapeHtml(opts.lab === undefined ? '' : opts.lab) + '">' +
      '<div class="verify-actions">' +
        '<button type="button" class="btn btn-primary btn-verify" data-action="verify">Verify</button>' +
        (opts.explore ? '' : '<a href="#" class="link-complete" data-action="complete">Complete this step</a>') +
      '</div>' +
      '<div class="verify-results" aria-live="polite"></div>' +
    '</div>';
  }

  function renderAnswer(text) {
    return '<details class="answer"><summary>Show answer</summary><div class="answer-body">' +
      renderInline(text) + '</div></details>';
  }

  // ------------------------------------------------------------------ blocks
  function splitRow(line) {
    let s = line.trim();
    if (s.charAt(0) === '|') s = s.slice(1);
    if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
    return s.split('|').map(function (cell) { return cell.trim(); });
  }

  const TABLE_SEPARATOR = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

  function renderTable(rows, aligns) {
    const head = rows[0];
    const cell = function (tag, text, i) {
      const align = aligns[i] ? ' style="text-align:' + aligns[i] + '"' : '';
      return '<' + tag + align + '>' + renderInline(text) + '</' + tag + '>';
    };
    let html = '<div class="table-wrap"><table class="md-table"><thead><tr>' +
      head.map(function (t, i) { return cell('th', t, i); }).join('') + '</tr></thead><tbody>';
    rows.slice(1).forEach(function (row) {
      html += '<tr>' + head.map(function (_t, i) { return cell('td', row[i] || '', i); }).join('') + '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function renderMarkdown(markdown, options) {
    const opts = options || {};
    const lines = String(markdown).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let para = [];
    let i = 0;
    let m;

    function flush() {
      if (para.length) {
        out.push('<p>' + renderInline(para.join(' ')) + '</p>');
        para = [];
      }
    }

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      if ((m = /^\s*```\s*([\w+-]*)\s*$/.exec(line))) {
        flush();
        const buf = [];
        i += 1;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i += 1; }
        i += 1;
        const body = m[1] === 'python' || m[1] === 'py' ? highlightPython(buf.join('\n')) : escapeHtml(buf.join('\n'));
        out.push('<pre class="code"><code>' + body + '</code></pre>');
        continue;
      }
      if (!trimmed) { flush(); i += 1; continue; }

      if ((m = /^@code\s+(\S+)$/.exec(trimmed))) { flush(); out.push(renderCodeCard(m[1], opts)); i += 1; continue; }
      if (/^@verify$/.test(trimmed)) { flush(); out.push(renderVerify(opts)); i += 1; continue; }
      if ((m = /^@answer\s+(.+)$/.exec(trimmed))) { flush(); out.push(renderAnswer(m[1])); i += 1; continue; }

      if ((m = /^(#{1,3})\s+(.+?)\s*#*$/.exec(trimmed))) {
        flush();
        const level = m[1].length;
        out.push('<h' + level + '>' + renderInline(m[2]) + '</h' + level + '>');
        i += 1;
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) { flush(); out.push('<hr>'); i += 1; continue; }

      if (trimmed.charAt(0) === '>') {
        flush();
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          buf.push(lines[i].replace(/^\s*>\s?/, ''));
          i += 1;
        }
        const inner = renderMarkdown(buf.join('\n'), Object.assign({}, opts, { nested: true }));
        out.push('<aside class="callout"><div class="callout-title">Tip</div>' + inner + '</aside>');
        continue;
      }

      if (trimmed.charAt(0) === '|' && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1])) {
        flush();
        const aligns = splitRow(lines[i + 1]).map(function (c) {
          if (/^:-+:$/.test(c)) return 'center';
          if (/^-+:$/.test(c)) return 'right';
          if (/^:-+$/.test(c)) return 'left';
          return '';
        });
        const rows = [splitRow(lines[i])];
        i += 2;
        while (i < lines.length && lines[i].trim().charAt(0) === '|') { rows.push(splitRow(lines[i])); i += 1; }
        out.push(renderTable(rows, aligns));
        continue;
      }

      if ((m = LIST_ITEM.exec(line))) {
        flush();
        const ordered = /\d/.test(m[2]);
        const start = ordered ? parseInt(m[2], 10) : 1;
        const items = [];
        while (i < lines.length) {
          const cur = lines[i];
          const lm = LIST_ITEM.exec(cur);
          if (lm && /\d/.test(lm[2]) === ordered) { items.push(lm[3]); i += 1; continue; }
          if (cur.trim() && /^\s+/.test(cur) && items.length && !lm) {
            items[items.length - 1] += ' ' + cur.trim();
            i += 1;
            continue;
          }
          break;
        }
        const tag = ordered ? 'ol' : 'ul';
        const startAttr = ordered && start !== 1 ? ' start="' + start + '"' : '';
        out.push('<' + tag + startAttr + '>' + items.map(function (t) { return '<li>' + renderInline(t) + '</li>'; }).join('') + '</' + tag + '>');
        continue;
      }

      para.push(trimmed);
      i += 1;
    }
    flush();
    return out.join('\n');
  }

  const api = {
    escapeHtml: escapeHtml,
    renderInline: renderInline,
    renderMarkdown: renderMarkdown,
    highlightPython: highlightPython,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MD = api;
})(typeof window !== 'undefined' ? window : this);
