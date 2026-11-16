# 5-Minute Demo Script

**Setup before presenting** (5 minutes ahead):
- [ ] `streamlit run src/app.py` is running, the browser window is wide (so the equalizer shows), and the zoom is at about 110%.
- [ ] A terminal is open in the project folder with `.venv` activated, for the CLI and test segments.
- [ ] Gemini quota: avoid heavy testing right before the demo. If Gemini is out, the app says so and the offline planner runs, so mention it as the fallback working.
- [ ] Spotify: `python -m src.export check` prints "works".
- [ ] Backup: the screenshots in `docs/images/` and the sample outputs in the README, in case the network fails.

---

## 0:00–0:30 · The problem and the pitch

> "Describing a vibe is easy. Finding songs that match it is not. *Give me your vibes, I give you a playlist*: you describe the music in plain English, an AI agent finds songs from a catalog of 16,000, checks its own picks, learns from your thumbs up and down, and gives you a playlist."

## 0:30–1:15 · Architecture (show the README diagram)

Point to the four sections:
- **Data (built offline):** the Spotify Tracks Dataset → 16,120 songs with embeddings.
- **The agent:** guardrail → plan (Gemini) → retrieve (embeddings + keywords) → rank → **self-check, retry** → reasons. In the refine round, **Gemini picks 5 of 30 candidates using your votes**, and the self-check checks its picks.
- **Human review:** 👍/👎 → one refine round.
- **Testing:** 132 tests and simulated users.

> "The AI features are RAG, since in the refine round Gemini chooses only among 30 songs we retrieved, an agentic loop that checks and fixes its own work, even Gemini's picks, and a reliability layer. Gemini handles understanding the request and judging taste from your votes; search, ranking and checking are plain Python, so they're testable and still work when the API is down."

## 1:15–3:15 · Live demo (the app)

1. **Start page:** point out the design (Figma first, the aura glows, the equalizer behind the content).
2. Click **"something chill, with jazz and indie music"**.
3. **Round 1:**
   - The plan chips: jazz, indie, chill, energy 0–0.45.
   - Play one Spotify preview for 3 seconds.
   - Read one reason: *"it's built only from that song's data."*
4. Vote: 👍 three songs, 👎 two.
5. **Refine my playlist:** the playlist so far plus 5 new songs.
   > "The search moved toward what I liked, and then Gemini read my 👍 and 👎 and chose these 5 out of 30 candidates. Look at the reasons: they mention the songs I voted on. The self-check still checks Gemini's picks, and no song repeats."
6. **Looks good, I'm done:** the final playlist.
   - **Get Spotify playlist link:** open it in Spotify.
   - **Download CSV.**

## 3:15–4:00 · Reliability (terminal)

```bash
python -m src.cli "something chill, with jazz and indie music" --offline --trace
```

> "Here's the agent checking its own work: attempt 1 had a song outside the energy target, so it made the target a hard filter and retried."

```bash
python -m src.cli "write my history essay about the french revolution" --offline
```

> "Off-topic requests are rejected before any search. Missing genres like k-pop are swapped for the closest one, with a note."

Then show the simulated-user table from the README:

> "Six simulated listeners with hidden tastes: after one round of votes, 70% of new songs are liked, versus 40% without feedback. That's why there's only one refine round: it captures most of the benefit and keeps us inside the free API quota."

## 4:00–4:45 · Limitations, ethics, what I learned

- **Mainstream bias:** the catalog favors popular, mostly Western songs.
- **Labels stand in for listening.**
- **Feedback narrows, it doesn't explore:** the latin listener stays at 0%.
- **Privacy:** no profiles are stored, and the app warns that request text goes to Gemini. Spotify never writes to a visitor's account.
- **Lesson:** use the LLM where language matters, keep the rest deterministic, and measure, because the simulator changed the design twice.

## 4:45–5:00 · Close

> "Give me your vibes, I give you a playlist. Thank you. Questions?"

---

**Likely questions:**
- *Is the feedback reinforcement learning?* No: it's relevance feedback (Rocchio). Only this session's search moves, no model is trained, and it resets after the session.
- *Why not fine-tune?* It was the highest effort and risk for the least benefit. Retrieval and rules covered the needs.
- *Does the equalizer react to the music?* No. Spotify's embedded player doesn't share its audio with the page, so it's decorative.
- *What if Gemini is down?* The keyword planner and template reasons take over for that step, and the app says so.
