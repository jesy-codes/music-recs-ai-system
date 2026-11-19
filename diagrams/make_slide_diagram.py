"""Build a 16:9 slide version of system_architecture.mmd.

Same content and colors as the Mermaid diagram, hand-laid out so it fills a
1920x1080 slide: the agent (purple) runs top to bottom in the middle, data is
on the left, human review and output on the right, testing across the top.

Run:  python3 diagrams/make_slide_diagram.py
Out:  diagrams/system_architecture_slide.svg
"""
import math
import sys
from pathlib import Path

OUT = Path(__file__).with_name("system_architecture_slide.svg")

INK = "#2C2C2A"
NODE_STROKE = "#888780"
FONT = "font-family=\"Inter, 'Helvetica Neue', Arial, sans-serif\""

parts = []


def add(s):
    parts.append(s)


def text(x, y, lines, size=17, weight="normal", color=INK, anchor="middle", lh=None):
    """Multi-line text, vertically centered on y."""
    lh = lh or size * 1.25
    y0 = y - (len(lines) - 1) * lh / 2 + size * 0.35
    spans = "".join(
        f'<tspan x="{x}" y="{y0 + i * lh:.1f}">{l}</tspan>' for i, l in enumerate(lines)
    )
    add(f'<text {FONT} font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{spans}</text>')


def section(x, y, w, h, title, fill, stroke, color):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>')
    text(x + w / 2, y + 26, [title], size=22, weight="bold", color=color)


def rect(cx, cy, lines, w=170, h=78, bold_first=True):
    add(f'<rect x="{cx - w/2}" y="{cy - h/2}" width="{w}" height="{h}" rx="8" '
        f'fill="#FFFFFF" stroke="{NODE_STROKE}" stroke-width="1.5"/>')
    _node_text(cx, cy, lines, bold_first)


def cylinder(cx, cy, lines, w=170, h=88):
    e = 10
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2 + e, cy + h / 2 - e
    add(f'<path d="M{x0},{y0} A{w/2},{e} 0 0 1 {x1},{y0} V{y1} A{w/2},{e} 0 0 1 {x0},{y1} Z" '
        f'fill="#FFFFFF" stroke="{NODE_STROKE}" stroke-width="1.5"/>')
    add(f'<path d="M{x0},{y0} A{w/2},{e} 0 0 0 {x1},{y0}" fill="none" '
        f'stroke="{NODE_STROKE}" stroke-width="1.5"/>')
    _node_text(cx, cy + 5, lines, True)


def diamond(cx, cy, lines, w=170, h=124):
    add(f'<polygon points="{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}" '
        f'fill="#FFFFFF" stroke="{NODE_STROKE}" stroke-width="1.5"/>')
    _node_text(cx, cy, lines, True, size=15)


def hexagon(cx, cy, lines, w=180, h=78):
    i = 22
    pts = [(cx - w/2, cy), (cx - w/2 + i, cy - h/2), (cx + w/2 - i, cy - h/2),
           (cx + w/2, cy), (cx + w/2 - i, cy + h/2), (cx - w/2 + i, cy + h/2)]
    add('<polygon points="' + " ".join(f"{x},{y}" for x, y in pts) +
        f'" fill="#FFFFFF" stroke="{NODE_STROKE}" stroke-width="1.5"/>')
    _node_text(cx, cy, lines, True)


def burst(cx, cy, r, title, lines, fill, stroke, color):
    pts = []
    for k in range(36):
        rr = r if k % 2 == 0 else r * 0.84
        a = math.pi * 2 * k / 36
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
    add(f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    text(cx, cy - 34, [title], size=24, weight="bold", color=color)
    text(cx, cy + 14, lines, size=15, color=color, lh=19)


def _node_text(cx, cy, lines, bold_first, size=16):
    lh = size * 1.25
    y0 = cy - (len(lines) - 1) * lh / 2 + size * 0.35
    spans = []
    for i, l in enumerate(lines):
        w = ' font-weight="bold"' if (i == 0 and bold_first) else ""
        spans.append(f'<tspan x="{cx}" y="{y0 + i * lh:.1f}"{w}>{l}</tspan>')
    add(f'<text {FONT} font-size="{size}" fill="{INK}" text-anchor="middle">{"".join(spans)}</text>')


def arrow(d, dashed=False, color="#444441", both=False):
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    start = ' marker-start="url(#arrowStart)"' if both else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash}{start} '
        f'marker-end="url(#arrow)"/>')


def label(x, y, lines, anchor="middle", size=14):
    """Edge label with a soft background so it stays readable over lines."""
    lh = size * 1.25
    w = max(len(l) for l in lines) * size * 0.56 + 12
    h = len(lines) * lh + 6
    bx = {"middle": x - w / 2, "start": x - 6, "end": x - w + 6}[anchor]
    add(f'<rect x="{bx:.1f}" y="{y - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
        f'fill="#FFFFFF" fill-opacity="0.92"/>')
    text(x, y, lines, size=size, color="#444441", anchor=anchor, lh=lh)


def build():
    add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" width="1920" height="1080">')
    add('<defs>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#444441"/></marker>'
        '<marker id="arrowStart" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#444441"/></marker>'
        '</defs>')
    add('<rect width="1920" height="1080" fill="#FFFFFF"/>')

    # Title
    text(40, 52, ["Agentic Playlist Builder · System Architecture"], size=34,
         weight="bold", anchor="start")

    # ---- Sections ----------------------------------------------------------
    section(40, 390, 260, 520, "Data, built offline", "#F1EFE8", "#5F5E5A", INK)
    section(340, 100, 660, 950, "Process: playlist agent", "#EEEDFE", "#534AB7", "#26215C")
    section(1050, 640, 450, 410, "Human review", "#FAEEDA", "#854F0B", "#412402")
    section(1050, 100, 830, 280, "Testing and evaluation", "#E1F5EE", "#0F6E56", "#04342C")

    # ---- Input / Output ----------------------------------------------------
    burst(160, 210, 120, "Input", ["user request:", "'something chill,", "with jazz and indie'"],
          "#FAC775", "#633806", "#412402")
    burst(1745, 945, 118, "Output", ["final playlist:", "liked songs only,", "CSV or Spotify link"],
          "#EAF3DE", "#3B6D11", "#173404")

    # ---- Data (left column, top to bottom) ---------------------------------
    cylinder(170, 490, ["Spotify Tracks", "Dataset, 114k rows"])
    rect(170, 640, ["Catalog builder", "dedupe, sample genres,", "vibe tags, embeddings"], w=200, h=90)
    cylinder(170, 800, ["Catalog", "16,120 songs +", "embeddings"], h=100)
    arrow("M170,534 V593")
    arrow("M170,685 V748")

    # ---- Agent (vertical spine at x=620) -----------------------------------
    X = 620
    diamond(X, 200, ["Input", "guardrail"], w=160, h=110)
    rect(X, 320, ["Planner", "Gemini: request", "to search query"])
    rect(X, 440, ["Retriever", "embeddings +", "keywords"])
    rect(X, 560, ["Ranker", "audio features +", "diversity"])
    diamond(X, 700, ["Self-check", "genre, energy,", "count, variety"], w=190, h=136)
    rect(850, 840, ["Gemini reranker", "refine round: picks", "5 of 30 using 👍/👎"], w=190, h=84)
    rect(X, 970, ["Reasons", "from song data", "only"])

    arrow(f"M{X},255 V279")
    label(X + 34, 267, ["valid"], anchor="start")
    arrow(f"M{X},359 V399")
    arrow(f"M{X},479 V519")
    arrow(f"M{X},599 V630")
    # self-check passes -> reasons
    arrow(f"M{X},768 V929")
    label(X - 14, 860, ["passes"], anchor="end")
    # self-check fails -> retriever (loop on the left)
    arrow(f"M{X - 95},700 H455 V455 H533")
    label(448, 590, ["fails: adjust,", "max 2 retries"], anchor="end")
    # refine round -> reranker -> reasons
    arrow(f"M{X + 95},700 H850 V796")
    label(785, 700, ["refine round"])
    arrow("M850,883 V955 H707")
    label(862, 920, ["self-check", "Gemini's picks"], anchor="start")

    # Input <-> guardrail
    arrow(f"M282,200 H{X - 82}", both=True)
    label(410, 160, ["request in;", "rejected: ask to rephrase"])

    # Data -> retriever
    arrow("M255,800 H320 V425 H533")

    # ---- Human review ------------------------------------------------------
    rect(1160, 970, ["5 proposed songs", "preview + reason"], w=170, h=70)
    hexagon(1385, 970, ["User votes", "👍 / 👎"])
    rect(1385, 760, ["Feedback update", "toward liked songs,", "away from rejected"], w=180, h=84)

    arrow(f"M{X + 87},985 H1073")
    arrow("M1245,970 H1293")
    arrow("M1385,931 V804")
    label(1395, 868, ["refine"], anchor="start")
    arrow("M1475,970 H1618")
    label(1546, 950, ["done"])
    # feedback -> retriever (playlist so far + 5 new songs)
    arrow(f"M1385,718 V440 H{X + 87}")
    label(1170, 440, ["playlist so far + 5 new songs"])

    # ---- Testing (top right) -----------------------------------------------
    rect(1190, 260, ["Unit tests", "pytest, Gemini and", "Spotify mocked"], w=190, h=84)
    rect(1430, 260, ["Evaluator", "golden prompts,", "precision@5, logs"], w=190, h=84)
    rect(1690, 260, ["Simulated user", "feedback lift", "per round"], w=190, h=84)
    arrow("M1095,260 H1004", dashed=True)
    arrow("M1430,302 V340 H1004", dashed=True)
    label(1030, 222, ["tests"], anchor="start")
    # simulated user -> votes, routed down the gap between human review and output
    arrow("M1690,302 V560 H1545 V900 H1445 V929", dashed=True)
    label(1560, 730, ["simulated", "votes"], anchor="start")

    # ---- Legend (bottom left) ----------------------------------------------
    add('<line x1="50" y1="960" x2="110" y2="960" stroke="#444441" stroke-width="2.2" marker-end="url(#arrow)"/>')
    text(122, 960, ["data flow"], size=16, anchor="start")
    add('<line x1="50" y1="1000" x2="110" y2="1000" stroke="#444441" stroke-width="2.2" '
        'stroke-dasharray="7 6" marker-end="url(#arrow)"/>')
    text(122, 1000, ["testing"], size=16, anchor="start")

    add("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    try:
        OUT.write_text(build(), encoding="utf-8")
    except OSError as e:
        print(f"error: could not write {OUT}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {OUT}")
