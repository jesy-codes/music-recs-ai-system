"""Visual layer for the Streamlit app: the "Aura blue" design from Figma.

Figma file: "Vibe Playlist Builder – UI", frames "v3 Aura blue – Start / Round / Final".
Colors and fonts mirror the Figma variables (Aura mode) and text styles. The theme basics (dark base,
Montserrat, accent blue) are in .streamlit/config.toml; this module adds what the theme can't express:
the drifting aura glows, frosted-glass song cards, gradient buttons, round 👍/👎 buttons and chips.

Every value that comes from data (song titles, artists, genres, reasons, the user's request) is
HTML-escaped before it is placed in markup.
"""

from __future__ import annotations

import math
from html import escape
from typing import Iterable, Sequence

# Figma "Aura" mode tokens
BG = "#06080D"
SURFACE = "rgba(255,255,255,0.05)"
BORDER = "rgba(255,255,255,0.10)"
TEXT = "#F4F2FA"
TEXT_SOFT = "#A19DB3"
ACCENT = "#3B82F6"        # blue
ACCENT_2 = "#93C5FD"      # light blue
ACCENT_3 = "#22D3EE"      # cyan
GREEN = "#10B981"         # emerald (aura transition)
DEEP_BLUE = "#1E40AF"
NEGATIVE = "#334155"      # slate (active 👎)

CSS = f"""
<style>
/* ---------- background: aura glows that slowly drift ---------- */
.stApp {{ background: {BG}; isolation: isolate; }}
.stApp::before {{
  content: ""; position: fixed; inset: -25%; z-index: -1; pointer-events: none;
  background:
    radial-gradient(38% 34% at 18% 12%, rgba(37,99,235,0.55), transparent 70%),
    radial-gradient(30% 28% at 40% 26%, rgba(16,185,129,0.30), transparent 70%),
    radial-gradient(30% 30% at 84% 10%, rgba(34,211,238,0.34), transparent 70%),
    radial-gradient(36% 32% at 72% 72%, rgba(30,64,175,0.26), transparent 72%),
    radial-gradient(28% 26% at 22% 82%, rgba(16,185,129,0.14), transparent 70%);
  filter: blur(40px);
  animation: aura-drift 22s ease-in-out infinite alternate;
}}
@keyframes aura-drift {{
  0%   {{ transform: translate3d(-3%, -2%, 0) scale(1.00) rotate(0deg); }}
  50%  {{ transform: translate3d(3%, 3%, 0)  scale(1.08) rotate(4deg); }}
  100% {{ transform: translate3d(-1%, 5%, 0) scale(1.03) rotate(-3deg); }}
}}
@media (prefers-reduced-motion: reduce) {{ .stApp::before {{ animation: none; }} }}
[data-testid="stHeader"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: transparent; }}
[data-testid="stDecoration"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
[data-testid="stSidebar"] {{ background: rgba(8,11,18,0.82); backdrop-filter: blur(20px); border-right: 1px solid {BORDER}; }}

/* ---------- hero ---------- */
.hero {{ margin: 0.5rem 0 1.25rem; }}
.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
  background: rgba(59,130,246,0.18); color: {ACCENT_2}; font-size: 13px; font-weight: 500; }}
.hero-title {{ font-size: clamp(2rem, 5.2vw, 2.75rem); font-weight: 700; line-height: 1.1; letter-spacing: -0.02em;
  color: {TEXT}; margin: 0.75rem 0 0.5rem; padding: 0; }}
.hero.compact .hero-title {{ font-size: clamp(1.5rem, 3.6vw, 2rem); }}
.hero-title .c1 {{ color: {ACCENT}; }}
.hero-title .c2 {{ color: {ACCENT_2}; }}
.hero-sub {{ color: {TEXT_SOFT}; font-size: 15px; margin: 0; }}
.muted {{ color: {TEXT_SOFT}; font-size: 13px; }}

/* ---------- chat rows ---------- */
.user-row {{ display: flex; justify-content: flex-end; margin: 0.25rem 0 0.75rem; }}
.user-bubble {{ max-width: 80%; padding: 10px 18px; border-radius: 20px 20px 6px 20px; background: rgba(59,130,246,0.18);
  border: 1px solid {ACCENT}; color: {TEXT}; font-weight: 600; font-size: 14px; }}
[data-testid="stChatMessage"] {{ background: transparent; padding-left: 0; padding-right: 0; }}
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {{ background: rgba(255,255,255,0.08); color: {TEXT};
  border-radius: 999px; }}
.assistant-title {{ font-size: 21px; font-weight: 600; letter-spacing: -0.01em; color: {TEXT}; margin: 0.15rem 0 0.5rem; }}

/* ---------- chips ---------- */
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }}
.chip {{ padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 500; background: rgba(255,255,255,0.08); color: {TEXT}; }}
.chip.accent {{ background: rgba(59,130,246,0.18); color: {ACCENT_2}; }}
.chip.cyan {{ background: rgba(34,211,238,0.14); color: {ACCENT_3}; }}
.chip.green {{ background: rgba(16,185,129,0.16); color: #6EE7B7; }}

/* ---------- song cards (st.container key="song_...") ---------- */
[class*="st-key-song_"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px; padding: 14px 16px;
  backdrop-filter: blur(24px); box-shadow: 0 8px 32px -4px rgba(59,130,246,0.16); margin-bottom: 4px; }}
.song-title {{ font-weight: 700; font-size: 16px; color: {TEXT}; line-height: 1.3; }}
.song-artist {{ color: {TEXT_SOFT}; font-size: 14px; margin-bottom: 2px; }}
.song-reason {{ color: {TEXT_SOFT}; font-size: 13px; line-height: 1.45; margin: 2px 0 4px; }}
.section-label {{ color: {TEXT_SOFT}; font-size: 13px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin: 0.75rem 0 0.25rem; }}
[class*="st-key-song_"] iframe {{ border-radius: 12px; }}

/* ---------- 👍 / 👎 (st.feedback key="vote_...") ---------- */
[class*="st-key-vote_"] button {{ width: 42px; min-width: 42px; height: 42px; padding: 0; margin-right: 8px; justify-content: center; border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.14) !important; background: rgba(255,255,255,0.05) !important; color: {TEXT_SOFT} !important; }}
[class*="st-key-vote_"] button:hover {{ border-color: {ACCENT_2} !important; color: {TEXT} !important; }}
[class*="st-key-vote_"] button[aria-label="Thumbs up"][aria-checked="true"] {{ background: {ACCENT} !important; border-color: {ACCENT} !important; color: #fff !important; }}
[class*="st-key-vote_"] button[aria-label="Thumbs down"][aria-checked="true"] {{ background: {NEGATIVE} !important; border-color: {NEGATIVE} !important; color: #fff !important; }}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {{ background: linear-gradient(90deg, {ACCENT}, {ACCENT_3}) !important; border: none !important;
  color: #fff !important; font-weight: 600; box-shadow: 0 6px 24px -6px rgba(34,211,238,0.45); }}
[data-testid="stBaseButton-primary"]:disabled {{ opacity: 0.45; box-shadow: none; }}
[data-testid="stBaseButton-secondary"] {{ background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.14) !important;
  color: {TEXT} !important; font-weight: 600; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: {ACCENT_2} !important; }}

/* ---------- chat input ---------- */
[data-testid="stChatInput"] {{ border-radius: 999px !important; border: 1.5px solid rgba(147,197,253,0.55) !important;
  background: rgba(255,255,255,0.05) !important; box-shadow: 0 8px 32px -6px rgba(59,130,246,0.30); }}
[data-testid="stChatInput"] textarea {{ color: {TEXT}; }}
[data-testid="stChatInputSubmitButton"] {{ background: {ACCENT} !important; border-radius: 999px !important; color: #fff !important; }}

/* ---------- final playlist ---------- */
.playlist-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px; padding: 14px 20px; margin: 4px 0 14px;
  backdrop-filter: blur(24px); box-shadow: 0 8px 32px -4px rgba(59,130,246,0.16); }}
.playlist-row {{ display: flex; align-items: center; gap: 14px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
.playlist-row:last-child {{ border-bottom: none; }}
.num {{ flex: none; width: 30px; height: 30px; border-radius: 999px; background: {ACCENT}; color: #fff; font-weight: 700;
  font-size: 13px; display: flex; align-items: center; justify-content: center; }}
.playlist-row .song {{ flex: 1; min-width: 0; }}
.playlist-row a {{ color: {ACCENT_2} !important; font-size: 13px; font-weight: 500; text-decoration: none; white-space: nowrap; }}
.banner {{ display: inline-block; padding: 4px 12px; border-radius: 999px; background: rgba(16,185,129,0.16); color: #6EE7B7;
  font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
/* ---------- left-side equalizer: horizontal bars that grow left -> right, behind the content ---------- */
.eq {{ position: fixed; left: 0; top: 0; bottom: 0; width: min(62vw, 980px); z-index: -1; pointer-events: none;
  display: flex; flex-direction: column; justify-content: space-between; padding: 6px 0;
  -webkit-mask-image: linear-gradient(180deg, transparent 0%, #000 5%, #000 95%, transparent 100%);
          mask-image: linear-gradient(180deg, transparent 0%, #000 5%, #000 95%, transparent 100%); opacity: 0.38; }}
.eq span {{ display: block; flex: none; height: 4px; width: 100%; border-radius: 0 999px 999px 0; transform-origin: left center;
  background: linear-gradient(90deg, rgba(59,130,246,0.95), rgba(16,185,129,0.75) 55%, rgba(34,211,238,0.0));
  transform: scaleX(var(--rest)); animation: eq-pulse var(--d) ease-in-out var(--delay) infinite alternate; }}
@keyframes eq-pulse {{
  0%   {{ transform: scaleX(var(--rest)); }}
  30%  {{ transform: scaleX(var(--peak)); }}
  55%  {{ transform: scaleX(calc(var(--rest) + 0.12)); }}
  80%  {{ transform: scaleX(calc(var(--peak) * 0.75)); }}
  100% {{ transform: scaleX(calc(var(--rest) * 1.4)); }}
}}
@media (max-width: 700px) {{ .eq {{ opacity: 0.25; }} }}
@media (prefers-reduced-motion: reduce) {{ .eq span {{ animation: none; transform: scaleX(var(--peak)); }} }}
</style>
"""

CHIP_CLASSES = ("accent", "", "cyan", "green")


def hero_html(compact: bool = False) -> str:
    sub = "" if compact else (
        '<p class="hero-sub">Describe a vibe. Vote 👍/👎 on the songs, get one refined round, and keep the playlist.</p>')
    return (
        f'<div class="hero{" compact" if compact else ""}">'
        '<span class="badge">♪ AI playlist builder</span>'
        '<h1 class="hero-title">🎵 Give me your <span class="c1">vibes</span>, I give you a <span class="c2">playlist</span>.</h1>'
        f"{sub}</div>"
    )


def chips_html(labels: Iterable[str], first_accent: bool = True) -> str:
    chips = []
    for i, label in enumerate(labels):
        cls = CHIP_CLASSES[i % len(CHIP_CLASSES)] if first_accent else ""
        chips.append(f'<span class="chip {cls}">{escape(label)}</span>')
    return f'<div class="chips">{"".join(chips)}</div>'


def song_info_html(title: str, artist: str, genres: str, reason: str) -> str:
    return (
        f'<div class="song-title">{escape(title)}</div>'
        f'<div class="song-artist">{escape(artist)}</div>'
        f'{chips_html([g for g in genres.split("; ") if g][:3])}'
        f'<div class="song-reason">{escape(reason)}</div>'
    )


def user_bubble_html(text: str) -> str:
    return f'<div class="user-row"><div class="user-bubble">{escape(text)}</div></div>'


def assistant_title_html(text: str) -> str:
    return f'<div class="assistant-title">{escape(text)}</div>'


def playlist_html(songs: Sequence[tuple[str, str, str]]) -> str:
    """songs: (title, artist, spotify_url)."""
    rows = "".join(
        f'<div class="playlist-row"><div class="num">{i}</div>'
        f'<div class="song"><div class="song-title">{escape(t)}</div><div class="song-artist">{escape(a)}</div></div>'
        f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">Open in Spotify ↗</a></div>'
        for i, (t, a, url) in enumerate(songs, start=1)
    )
    n = len(songs)
    return (f'<div class="playlist-card"><span class="banner">{n} song{"s" if n != 1 else ""} you gave a 👍</span>'
            f"{rows}</div>")


def visualizer_html(bars: int = 84) -> str:
    """Decorative equalizer on the left edge, top to bottom: horizontal bars pulsing left to right.

    Each bar gets its own speed, delay and peak length (deterministic, so the page renders the same way
    every time). It is pure CSS: it doesn't hear the music, since Spotify's embedded player doesn't
    expose its audio to the page.
    """
    spans = []
    for i in range(bars):
        # a slow wave down the page (two overlapping sines) so lengths vary from top to bottom
        wave = 0.5 + 0.25 * math.sin(i * 0.23) + 0.25 * math.sin(i * 0.071 + 1.3)
        peak = 0.3 + 0.62 * wave * (0.75 + 0.25 * ((i * 7) % 5) / 4)
        rest = 0.08 + 0.12 * ((i * 3) % 4) / 3
        duration = 0.9 + ((i * 13) % 9) * 0.17
        delay = -((i * 37) % 23) * 0.11
        spans.append(f'<span style="--peak:{peak:.2f};--rest:{rest:.2f};--d:{duration:.2f}s;--delay:{delay:.2f}s"></span>')
    return f'<div class="eq" aria-hidden="true">{"".join(spans)}</div>'
