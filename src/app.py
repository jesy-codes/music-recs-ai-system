"""Streamlit app: describe a vibe, vote 👍/👎, get one refined round, export the playlist.

Run:
    streamlit run src/app.py

Flow:
    1. Chat box -> round 1: 5 songs with Spotify previews and 👍/👎.
    2. "Refine my playlist" -> your playlist so far (👍 songs) + 5 new songs (one refine round).
    3. "Looks good, I'm done" -> final playlist (👍 songs only) -> CSV download / Spotify share link.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # `streamlit run src/app.py` puts src/ on the path

import streamlit as st

from src.agent import AgentResult, PlaylistAgent
from src.catalog import CatalogError
from src.export import ExportError, create_spotify_playlist, csv_filename, spotify_configured, to_csv_bytes
from src.llm import get_client
from src.log_config import setup_logging
from src.ranker import Recommendation
from src.retriever import Retriever
from src.session import PlaylistSession, SessionError
from src.styles import (CSS, assistant_title_html, chips_html, hero_html, playlist_html, song_info_html,
                        user_bubble_html, visualizer_html)

logger = logging.getLogger(__name__)

MAX_GEMINI_SESSIONS_PER_VISIT = 10   # after this many requests in one browser session, run offline (protects quota)
SECRET_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
               "SPOTIFY_REFRESH_TOKEN", "MAX_REFINE_ROUNDS")
EXAMPLES = ["something chill, with jazz and indie music", "music for a rainy sunday morning with coffee",
            "pump me up for the gym but no rap"]


# --- setup ----------------------------------------------------------------------

def load_secrets_into_env() -> None:
    """On Streamlit Cloud, keys live in st.secrets; locally they come from .env."""
    try:
        for key in SECRET_KEYS:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:   # no secrets.toml locally: that's fine
        pass


@st.cache_resource(show_spinner="Loading the song catalog…")
def load_agents() -> tuple[PlaylistAgent, PlaylistAgent, bool]:
    """(agent with Gemini if configured, offline agent, whether Gemini is on). Shared across visitors."""
    retriever = Retriever.load()
    llm = get_client()
    return PlaylistAgent(retriever, llm), PlaylistAgent(retriever, None), llm is not None


def init_state() -> None:
    defaults = {"session": None, "request": "", "done": False, "requests_made": 0,
                "spotify_link": None, "export_error": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset() -> None:
    for key in list(st.session_state):
        if key.startswith("vote_"):
            del st.session_state[key]
    st.session_state.update(session=None, request="", done=False, spotify_link=None, export_error=None)


# --- votes ------------------------------------------------------------------------

def vote_key(round_index: int, song_id: int) -> str:
    return f"vote_{round_index}_{song_id}"


def collect_votes(session: PlaylistSession) -> dict[int, bool]:
    """Read the 👍/👎 widgets of the latest round. 1 = 👍, 0 = 👎, None = no vote (keeps an earlier 👍)."""
    latest = len(session.rounds) - 1
    votes: dict[int, bool] = {}
    for song_id in session.shown:
        value = st.session_state.get(vote_key(latest, song_id))
        if value is not None:
            votes[song_id] = value == 1
    return votes


# --- rendering ------------------------------------------------------------------

def song_row(song: Recommendation, reason: str, key: str, default: int | None, active: bool) -> None:
    """One frosted-glass song card: Spotify preview on the left, info and 👍/👎 on the right."""
    with st.container(key=f"song_{key}"):
        preview, info = st.columns([1.15, 1], vertical_alignment="center", gap="medium")
        with preview:
            st.iframe(f"https://open.spotify.com/embed/track/{song.track_id}?utm_source=generator&theme=0", height=80)
        with info:
            st.markdown(song_info_html(song.title, song.artist, song.genres, reason), unsafe_allow_html=True)
            st.feedback("thumbs", key=key, default=default, disabled=not active)


def plan_chips(result: AgentResult) -> str:
    query = result.query
    labels = list(query.genres) + list(query.moods[:2])
    if query.energy:
        labels.append(f"energy {query.energy.low:.2f}–{query.energy.high:.2f}")
    if query.valence:
        labels.append(f"mood {query.valence.low:.2f}–{query.valence.high:.2f}")
    return chips_html(labels or ["open search"])


def render_round(session: PlaylistSession, index: int, result: AgentResult) -> None:
    latest = index == len(session.rounds) - 1 and not st.session_state.done
    with st.chat_message("assistant", avatar=":material/music_note:"):
        if result.status in ("rejected", "empty"):
            st.warning(result.message)
            return
        if result.status == "partial":
            st.warning(result.message)
        else:
            st.markdown(assistant_title_html(result.message), unsafe_allow_html=True)
        if index == 0 and result.query is not None:
            st.markdown(plan_chips(result), unsafe_allow_html=True)
            if session.agent.llm is not None and result.plan_source == "rules":
                st.caption(":orange[Gemini wasn't available for this request (quota or error), so the offline "
                           "planner was used. Results still come from the same catalog.]")
            for note in result.query.notes:
                st.info(note, icon="ℹ️")

        if index > 0:
            carried = [s for s in session.rounds_liked_before(index)]
            if carried:
                st.markdown(f'<div class="section-label">Your playlist so far ({len(carried)})</div>', unsafe_allow_html=True)
                for song in carried:
                    current = 0 if song.song_id in session.rejected else 1
                    song_row(song, session.reasons_by_id.get(song.song_id, "; ".join(song.reasons)),
                             vote_key(index, song.song_id), current, latest)
                st.markdown('<div class="section-label">New suggestions</div>', unsafe_allow_html=True)

        for song in result.songs:
            if latest:
                default = None
            else:
                default = 1 if song.song_id in session.liked else (0 if song.song_id in session.rejected else None)
            song_row(song, result.reasons.get(song.song_id, "; ".join(song.reasons)),
                     vote_key(index, song.song_id), default, latest)


def render_final(session: PlaylistSession) -> None:
    songs = session.playlist()
    with st.chat_message("assistant", avatar=":material/music_note:"):
        if not songs:
            st.warning("Your playlist is empty: no song got a 👍. Start over with a new request.")
            return
        st.markdown(assistant_title_html("Your playlist is ready!"), unsafe_allow_html=True)
        st.markdown(playlist_html([(s.title, s.artist, s.spotify_url) for s in songs]), unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.download_button("Download CSV", data=to_csv_bytes(songs), file_name=csv_filename(), type="primary",
                               mime="text/csv", icon=":material/download:", width="stretch")
        with right:
            if st.session_state.spotify_link:
                st.link_button("Open Spotify playlist", st.session_state.spotify_link,
                               icon=":material/open_in_new:", width="stretch")
            elif spotify_configured():
                if st.button("Get Spotify playlist link", icon=":material/queue_music:", width="stretch"):
                    try:
                        with st.spinner("Creating the playlist on Spotify…"):
                            st.session_state.spotify_link = create_spotify_playlist(songs)
                    except ExportError as error:
                        st.session_state.export_error = str(error)
                    st.rerun()
            else:
                st.button("Get Spotify playlist link", disabled=True, width="stretch",
                          help="Spotify export isn't configured on this server. Use the CSV or the song links.")
        if st.session_state.export_error:
            st.error(st.session_state.export_error)
        st.caption("The Spotify link points to a public playlist in the app owner's account; open it and save it "
                   "to your library if you like it. Nothing is added to your account.")
        if st.button("Start over", icon=":material/restart_alt:"):
            reset()
            st.rerun()


def render_actions(session: PlaylistSession) -> None:
    votes = collect_votes(session)
    new_votes = [i for i in votes if i in {s.song_id for s in session.rounds[-1].songs}]
    left, right = st.columns(2)
    with left:
        if session.can_refine:
            if st.button("Refine my playlist", type="primary", icon=":material/autorenew:", width="stretch",
                         disabled=not new_votes, help=None if new_votes else "Vote on at least one song first."):
                try:
                    with st.spinner("Learning from your votes and finding 5 more…"):
                        session.refine(votes)
                except SessionError as error:
                    st.error(str(error))
                st.rerun()
        else:
            st.caption("That's the last refine round. Vote on the new songs, then finish.")
    with right:
        if st.button("Looks good, I'm done", icon=":material/check:", width="stretch"):
            try:
                session.vote(votes)
            except SessionError as error:
                st.error(str(error))
            st.session_state.done = True
            st.rerun()


# --- page -------------------------------------------------------------------------

def start_session(request: str, agents: tuple[PlaylistAgent, PlaylistAgent, bool]) -> None:
    agent_llm, agent_offline, gemini_on = agents
    use_gemini = gemini_on and st.session_state.requests_made < MAX_GEMINI_SESSIONS_PER_VISIT
    reset()
    session = PlaylistSession(agent_llm if use_gemini else agent_offline, k=5)
    with st.spinner("Planning, searching and checking…"):
        session.start(request)
    st.session_state.update(session=session, request=request, requests_made=st.session_state.requests_made + 1)


def main() -> None:
    setup_logging(logging.WARNING)
    st.set_page_config(page_title="Vibe Playlist Builder", page_icon="🎵", layout="centered")
    load_secrets_into_env()
    init_state()

    st.html(CSS)
    st.markdown(visualizer_html(), unsafe_allow_html=True)
    st.markdown(hero_html(compact=st.session_state.session is not None), unsafe_allow_html=True)
    try:
        agents = load_agents()
    except CatalogError as error:
        st.error(f"The song catalog couldn't be loaded: {error}")
        st.stop()
    gemini_on = agents[2]

    session: PlaylistSession | None = st.session_state.session
    if session is None:
        st.markdown('<p class="muted">Try one of these, or type your own below:</p>', unsafe_allow_html=True)
        with st.container(horizontal=True, wrap=True, gap="small"):
            for example in EXAMPLES:
                if st.button(example, width="content"):
                    start_session(example, agents)
                    st.rerun()
        if gemini_on:
            st.markdown('<p class="muted">Requests are sent to Google\'s Gemini API to plan the search and write '
                        'the explanations, so don\'t include personal information.</p>', unsafe_allow_html=True)
    else:
        st.markdown(user_bubble_html(st.session_state.request), unsafe_allow_html=True)
        for index, result in enumerate(session.rounds):
            render_round(session, index, result)
        ok = session.rounds and session.rounds[-1].status in ("ok", "partial")
        if st.session_state.done:
            render_final(session)
        elif ok:
            render_actions(session)

    request = st.chat_input("Describe the music you want…", max_chars=500)
    if request:
        start_session(request, agents)
        st.rerun()


main()
