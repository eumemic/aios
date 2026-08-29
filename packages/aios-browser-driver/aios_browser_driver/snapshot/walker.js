// The aios snapshot walker — authored for this driver, no third-party code.
//
// Injected fresh on every snapshot via page.evaluate. Walks the MAIN frame's
// DOM (iframes render as unref'd items; their content is not walked — v1),
// descending open shadow roots, and collects the interactive elements and
// headings a model can act on. Each collected element gets a [ref=eN] handle
// registered in the page-scoped global `window.__aiosRefs`, which holds only
// the CURRENT generation's map (`{gen, map}`) — a newer snapshot overwrites it
// wholesale, and navigation destroys the document (and the registry with it),
// so a ref never resolves outside the snapshot that minted it. The driver's
// own `issued` watermark is what separates "was issued, now superseded"
// (stale_snapshot) from "never issued" (no_such_ref); the page side keeps no
// history.
//
// Credential safety: this runs in the PAGE's (hostile) realm, so `el.type` can
// be shadowed by page JS. The password check therefore folds in the type
// ATTRIBUTE and autocomplete before emitting any input value — best-effort
// masking; the real guarantee is that credential-bearing fields are the
// owner's, entered under a human takeover, never typed by the agent.
//
// Input:  [gen, startRef, maxElements]
// Output: {items: [...], assigned: N, omitted: M}
//   item: {role, name, tag, ref?, level?, value?, checked?, disabled?,
//          expanded?, options?}
([gen, startRef, maxElements]) => {
  const SKIP_SUBTREE = new Set(["script", "style", "noscript", "template", "select", "datalist", "svg", "iframe", "object", "embed"]);
  const INPUT_ROLES = {
    checkbox: "checkbox",
    radio: "radio",
    range: "slider",
    button: "button",
    submit: "button",
    reset: "button",
    image: "button",
    file: "button",
    search: "searchbox",
    color: "button",
  };
  const TAG_ROLES = { button: "button", textarea: "textbox", summary: "button" };
  const NAME_MAX = 80;
  const VALUE_MAX = 60;
  const OPTIONS_MAX = 12;

  const clip = (text, max) => {
    const collapsed = (text || "").replace(/\s+/g, " ").trim();
    return collapsed.length > max ? collapsed.slice(0, max - 1) + "…" : collapsed;
  };

  // Best-effort in a hostile realm: any of the three signals marks the field
  // credential-bearing, so its value is never emitted as a value OR a name.
  const isPasswordish = (el) =>
    el instanceof HTMLInputElement &&
    (el.type === "password" ||
      (el.getAttribute("type") || "").toLowerCase() === "password" ||
      (el.getAttribute("autocomplete") || "").toLowerCase().includes("password"));

  const accName = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria && aria.trim()) return clip(aria, NAME_MAX);
    const labelledby = el.getAttribute("aria-labelledby");
    if (labelledby) {
      const text = labelledby
        .split(/\s+/)
        .map((id) => (el.getRootNode().getElementById?.(id)?.textContent || ""))
        .join(" ");
      if (text.trim()) return clip(text, NAME_MAX);
    }
    if (el.labels && el.labels.length) {
      const text = Array.from(el.labels).map((l) => l.textContent || "").join(" ");
      if (text.trim()) return clip(text, NAME_MAX);
    }
    for (const attr of ["alt", "title", "placeholder"]) {
      const v = el.getAttribute(attr);
      if (v && v.trim()) return clip(v, NAME_MAX);
    }
    if (
      el instanceof HTMLInputElement &&
      (el.type === "submit" || el.type === "reset" || el.type === "button") &&
      el.value &&
      !isPasswordish(el)
    ) {
      return clip(el.value, NAME_MAX);
    }
    return clip(el.textContent || "", NAME_MAX);
  };

  const roleOf = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit.trim().split(/\s+/)[0].toLowerCase();
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return el.hasAttribute("href") ? "link" : null;
    if (tag === "input") {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      if (type === "hidden") return null;
      return INPUT_ROLES[type] || "textbox";
    }
    if (tag === "select") return el.multiple ? "listbox" : "combobox";
    if (/^h[1-6]$/.test(tag)) return "heading";
    if (TAG_ROLES[tag]) return TAG_ROLES[tag];
    if (el.isContentEditable && el.contentEditable === "true") return "textbox";
    return null;
  };

  const isInteractive = (el, role) => {
    if (role !== "generic" && role !== "presentation" && role !== "none") return true;
    if (el.hasAttribute("onclick")) return true;
    const tabindex = el.getAttribute("tabindex");
    return tabindex !== null && Number(tabindex) >= 0;
  };

  const visible = (el) => {
    try {
      if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false;
    } catch {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const items = [];
  let next = startRef;
  let omitted = 0;
  const refmap = {};

  const collect = (el) => {
    const tag = el.tagName.toLowerCase();

    if (tag === "iframe" || tag === "frame") {
      if (items.length < maxElements) {
        items.push({ role: "iframe", name: accName(el), tag });
      } else {
        omitted += 1;
      }
      return;
    }

    const role = roleOf(el);
    const heading = role === "heading";
    if (role === null || (!heading && !isInteractive(el, role))) return;
    if (!visible(el)) return;

    if (items.length >= maxElements) {
      omitted += 1;
      return;
    }

    const item = { role, name: accName(el), tag };
    const ref = "e" + next;
    next += 1;
    refmap[ref] = el;
    item.ref = ref;

    if (heading) {
      const m = tag.match(/^h([1-6])$/);
      item.level = m ? Number(m[1]) : Number(el.getAttribute("aria-level")) || 2;
    }
    if (el.disabled === true || el.getAttribute("aria-disabled") === "true") item.disabled = true;
    const expanded = el.getAttribute("aria-expanded");
    if (expanded === "true" || expanded === "false") item.expanded = expanded === "true";
    if (el instanceof HTMLInputElement) {
      if (el.type === "checkbox" || el.type === "radio") item.checked = el.checked;
      // Never a credential value (see isPasswordish — hostile-realm safe).
      else if (!isPasswordish(el) && el.value) item.value = clip(el.value, VALUE_MAX);
    } else if (el instanceof HTMLTextAreaElement && el.value) {
      item.value = clip(el.value, VALUE_MAX);
    } else if (el instanceof HTMLSelectElement) {
      item.options = Array.from(el.options)
        .slice(0, OPTIONS_MAX)
        .map((o) => ({ value: o.value, label: clip(o.label || o.textContent, VALUE_MAX), selected: o.selected }));
      if (el.options.length > OPTIONS_MAX) item.optionsOmitted = el.options.length - OPTIONS_MAX;
    }
    items.push(item);
  };

  // Bound recursion: a hostile page can nest the DOM thousands deep, which
  // would blow the JS stack and turn every snapshot into an `internal` error.
  const MAX_DEPTH = 200;
  const walk = (root, depth) => {
    if (depth > MAX_DEPTH) return;
    for (const el of root.children) {
      const tag = el.tagName.toLowerCase();
      if (el.getAttribute("aria-hidden") === "true") continue;
      collect(el);
      if (SKIP_SUBTREE.has(tag)) continue;
      if (el.shadowRoot) walk(el.shadowRoot, depth + 1);
      walk(el, depth + 1);
    }
  };

  walk(document.body || document.documentElement, 0);

  window.__aiosRefs = { gen, map: refmap };
  return { items, assigned: next - startRef, omitted };
}
