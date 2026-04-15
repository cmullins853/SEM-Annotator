from nicegui import ui

# ── Color constants (importable by other modules) ──────────────────────────────
# Change values here to retheme the entire app.

# Brand / interactive  (blue family)
PRIMARY        = '#60a5fa'   # blue-400 — buttons, toggles, accents
PRIMARY_DIM    = '#3b82f6'   # blue-500 — dimmer interactive text
PRIMARY_LIGHT  = '#bfdbfe'   # blue-200 — lightest interactive text
SECONDARY      = '#1e3a5f'   # dark blue — secondary interactive
ACCENT         = '#a78bfa'   # violet-400

# Quasar state colors
POSITIVE       = '#22c55e'
NEGATIVE       = '#ef4444'
INFO           = '#22d3ee'
WARNING        = '#f59e0b'   # amber-500

# Surfaces / backgrounds  (cool slate)
PAGE_BG        = '#0f1114'   # near-black, slight cool tint
DARK           = '#1a1d23'   # dark slate
SURFACE        = '#1e2128'   # sidebar, header, cards
MAIN_BG        = '#15181e'   # center content area

# Text
TEXT_NORMAL    = '#d1d5db'   # gray-300
TEXT_MUTED     = '#9ca3af'   # gray-400  — labels, hints
TEXT_SUBTLE    = '#6b7280'   # gray-500  — secondary hints
TEXT_FAINT     = '#4b5563'   # gray-600  — placeholder / empty states

# Warning / amber
WARN_BG        = '#1c0f00'   # dark amber — banner bg
WARN_BORDER    = '#92400e'   # amber-800 — banner border
WARN_TEXT      = '#fde68a'   # amber-200 — banner body text
WARN_ICON      = '#fbbf24'   # amber-400 — warning icon
WARN_STATUS_BG = '#78350f'   # amber-900 — status badge bg
WARN_STATUS_TXT= '#fcd34d'   # amber-300 — status badge text

# OK / emerald
OK_STATUS_BG   = '#064e3b'   # emerald-900 — status badge bg
OK_STATUS_TXT  = '#6ee7b7'   # emerald-300 — status badge text

# Drawing-tip banner  (blue family)
TIP_BG         = '#172554'   # blue-950
TIP_BORDER     = '#2563eb'   # blue-600
TIP_TEXT       = '#bfdbfe'   # blue-200

# Borders
BORDER_PRIMARY = 'rgba(96, 165, 250, 0.2)'   # blue-400 at 20% opacity
BORDER_MUTED   = '#374151'                    # gray-700

# SVG overlay colors (used in image_viewer SVG generation)
SCALE_COLOR    = '#00ff88'   # scale-line draw overlay
ROI_COLOR      = NEGATIVE    # ROI exclusion overlay — reuse negative


def apply_theme() -> None:
    """Apply FiberSight dark theme — Quasar dark mode + teal accent.

    All colors are defined as module-level constants above.
    Import constants directly for use outside of CSS (e.g., SVG generation).
    Use the CSS utility classes (fs-*) for NiceGUI widget .classes() calls.
    Use Quasar semantic names (primary, accent, positive …) for props color=.
    """
    ui.colors(
        primary=PRIMARY,
        secondary=SECONDARY,
        accent=ACCENT,
        dark=DARK,
        dark_page=PAGE_BG,
        positive=POSITIVE,
        negative=NEGATIVE,
        info=INFO,
        warning=WARNING,
    )
    ui.add_css(f'''
        /* ── Log panel ─────────────────────────────────────────────────── */
        .nicegui-log {{
            background-color: #0d1117 !important;
            font-family: "Courier New", Courier, monospace !important;
            font-size: 12px !important;
            color: #c9d1d9 !important;
            padding: 8px !important;
            border-radius: 4px;
        }}
        .fs-sidebar-card .q-expansion-item__content {{
            padding: 0 8px 8px 8px;
        }}

        /* ── Semantic text colors ───────────────────────────────────────── */
        .fs-text-primary      {{ color: {PRIMARY}; }}
        .fs-text-primary-dim  {{ color: {PRIMARY_DIM}; }}
        .fs-text-primary-lt   {{ color: {PRIMARY_LIGHT}; }}
        .fs-text-normal       {{ color: {TEXT_NORMAL}; }}
        .fs-text-muted        {{ color: {TEXT_MUTED}; }}
        .fs-text-subtle       {{ color: {TEXT_SUBTLE}; }}
        .fs-text-faint        {{ color: {TEXT_FAINT}; }}
        .fs-text-warn         {{ color: {WARN_TEXT}; }}
        .fs-text-warn-icon    {{ color: {WARN_ICON}; }}
        .fs-text-warn-status  {{ color: {WARN_STATUS_TXT}; }}
        .fs-text-ok           {{ color: {OK_STATUS_TXT}; }}
        .fs-text-tip          {{ color: {TIP_TEXT}; }}

        /* ── Semantic background colors ─────────────────────────────────── */
        .fs-bg-surface        {{ background-color: {SURFACE}; }}
        .fs-bg-main           {{ background-color: {MAIN_BG}; }}
        .fs-bg-warn           {{ background-color: {WARN_BG}; }}
        .fs-bg-warn-status    {{ background-color: {WARN_STATUS_BG}; }}
        .fs-bg-ok-status      {{ background-color: {OK_STATUS_BG}; }}
        .fs-bg-tip            {{ background-color: {TIP_BG}; }}

        /* ── Semantic border colors ──────────────────────────────────────── */
        .fs-border-primary    {{ border-color: {BORDER_PRIMARY}; }}
        .fs-border-muted      {{ border-color: {BORDER_MUTED}; }}
        .fs-border-warn       {{ border-color: {WARN_BORDER}; }}
        .fs-border-tip        {{ border-color: {TIP_BORDER}; }}
    ''')
