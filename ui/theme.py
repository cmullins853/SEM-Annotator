from nicegui import ui


def apply_theme() -> None:
    """Apply FiberSight dark theme — Quasar dark mode + teal accent."""
    ui.colors(
        primary='#2dd4bf',
        secondary='#1e3a5f',
        accent='#2dd4bf',
        dark='#16213e',
        dark_page='#1a1a2e',
        positive='#4caf50',
        negative='#f44336',
        info='#2196f3',
        warning='#f59e0b',
    )
    ui.add_css('''
        .nicegui-log {
            background-color: #0d1117 !important;
            font-family: "Courier New", Courier, monospace !important;
            font-size: 12px !important;
            color: #c9d1d9 !important;
            padding: 8px !important;
            border-radius: 4px;
        }
        .fs-sidebar-card .q-expansion-item__content {
            padding: 0 8px 8px 8px;
        }
    ''')
