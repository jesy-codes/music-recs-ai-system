# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## Setup

Requires Python 3.10+ (tested on 3.14, including Intel Macs).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests (no network or API keys needed):

```bash
python -m pytest
```

## Song catalog and embeddings

The repo already includes the built catalog, so you only need this step to rebuild it or change the settings.

```bash
python -m src.catalog build                    # download raw data if missing, build CSV + embeddings
python -m src.catalog build --per-genre 80     # bigger catalog
python -m src.catalog build --skip-embeddings  # CSV only
python -m src.catalog build --force-download   # fetch the raw dataset again
```

What it does:

1. **Downloads** the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (114,000 rows, BSD license) to `data/raw/` (not committed).
2. **Merges duplicates.** The raw data repeats a song once per genre and often under several track IDs; these are merged into one row per song with all its genres (81,206 unique songs).
3. **Samples** the 50 most popular songs from each of 30 genres (jazz, indie, chill, acoustic, rock, metal, …) → **1,500 songs** in `data/songs.csv`. Genres outside the list, such as k-pop, are left out on purpose to test the "not in catalog" guardrail.
4. **Adds vibe tags** from audio features, e.g. energy < 0.4 → "calm, mellow, chill"; acousticness > 0.6 → "acoustic, organic" (rules in `vibe_tags()` in `src/catalog.py`).
5. **Embeds** one sentence per song ("Blue Moon by Billie Holiday. Genres: blues, jazz. Album: … Feels: calm, mellow, chill, acoustic.") with the static model [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) → `data/embeddings.npy` (1,500 × 256). Static embeddings run on numpy alone, so no PyTorch is needed. That matters because PyTorch has no builds for Intel Macs on current Python versions, and it keeps the cloud deployment small. Override the model with the `EMBEDDING_MODEL` environment variable.
6. **Records** the model name and a hash of `songs.csv` in `data/catalog_meta.json`. If the CSV changes without rebuilding the embeddings, loading fails with a clear error instead of returning wrong results.

Logs go to the console and `logs/app.log`. Common errors and fixes:

| Error | Fix |
|---|---|
| `Couldn't download the dataset` | Check your connection, or download `dataset.csv` from the link above and save it as `data/raw/spotify_tracks.csv` |
| `Couldn't load embedding model` | The first build needs internet to download the model (~30 MB); later runs use the cache |
| `data/songs.csv changed since the embeddings were built` | Run `python -m src.catalog build` |

> The sections below describe the earlier profile-based version and will be rewritten for the agent.

---

## How The System Works

Real-world recommendations are based on 2 different methodologies: content-based (how similar the song is to the songs recommended) and collaborative-filtering (finding users with similar songs). 
Music and video platforms use a mix of these methodologies to recommend their users songs, videos (even products to buy).
My version of music recommender will skew towards content-based filtering, because I think it gives a bit of a priority to local and unknown artistics over popular ones. 

How this system works?
- In this simulation, the feautres used are `genre`, `mood`, `energy`, `valence` and `danceability`.
What does information `UserProfile store?
- I think initially, favorite genre, second favorite genre, mood at morning, mood at night, target energy. 
How does your `Recommender` compute a score for each song?
Each song earns 1.5 points for matching the favorite genre or 0.75 for matching the second favorite. It earns 1.5 points if its mood matches the user's mood for the current time of day. It also earns up to 2.0 points based on how close its energy is to the target, calculated as 2 × (1 − |song energy − target energy|). The song's score is the total of these points.
How do you choose which songs to recommend?
Score every song, sort from highest to lowest, and return the top k, with two small rules added:

1. Score: run every song in the catalog through the scoring rule for the current time of day.
2. Sort: order the songs from highest score to lowest.
3. Break ties: if two songs have the same score, put the one closer to the target energy first, then go alphabetically by title. This keeps the results the same on every run.
4. Return: take the top k songs, with a reason for each one.

```text
INPUT
  User prefs                               data/songs.csv
  ├─ favorite genre                              │
  ├─ second favorite genre                  load_songs()
  ├─ morning mood                                │
  ├─ night mood                                  │
  ├─ target energy                               │
  └─ time of day (morning/night)                 │
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
PROCESS: repeat for every song
  ┌──────────────────────────────────────────────┐
  │ score_song(user_prefs, song)                 │
  │                                              │
  │   Genre:  favorite +1.5, second +0.75        │
  │   Mood:   matches current mood +1.5          │
  │   Energy: + 2 × (1 − |song energy − target|) │
  │                                              │
  │   Returns: score and reasons                 │
  └──────────────────────┬───────────────────────┘
                         ▼
             Save (song, score, reasons)
                         │
                         ▼
OUTPUT: recommend_songs()
  1. Sort by score, highest first
  2. Break ties: closer energy, then title
  3. Keep the top k
                         │
                         ▼
  Top k recommendations: song, score, explanation
```

**Potential Biases**

- **Labels over sound:** genre and mood together are worth up to 3.0 of the 5.0 possible points. A song that misses both can earn at most 2.0, even if its energy is a perfect match.
- **Exact label matching:** `pop` and `indie pop` count as different genres, and so do `chill` and `relaxed`. Songs with a close but different label lose points, even if they sound similar.
- **Uneven catalog:** some genres have several songs and others have only one, and a genre missing from the catalog can never earn genre points. Users with those tastes get weaker recommendations.
- **Filter bubble:** the system only rewards what the user already likes and never tries anything new, so they may see the same few songs and rarely discover other genres.
- **One energy target all day:** a single target can penalize songs that fit the mood. A high target works against quiet, melancholy songs at night.
- **Alphabetical tie-breaks:** when songs tie on both score and energy, titles earlier in the alphabet always rank first.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Output of `python -m src.main` for a user whose favorite genre is rock, second favorite is metal, morning mood is intense, night mood is melancholy, and target energy is 0.85, with the time of day set to morning:

```text
Loading songs from data/songs.csv...
Loaded 30 songs from the dataset.

Top 5 recommendations (morning)
============================================
1. Storm Runner                 Score: 4.88
   - genre match (+1.5)
   - morning mood match (+1.5)
   - energy closeness (+1.88)

2. Gym Hero                     Score: 3.34
   - morning mood match (+1.5)
   - energy closeness (+1.84)

3. Iron Cathedral               Score: 2.51
   - second favorite genre match (+0.75)
   - energy closeness (+1.76)

4. Banjo Breakneck              Score: 1.96
   - energy closeness (+1.96)

5. Afterglow Protocol           Score: 1.94
   - energy closeness (+1.94)
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried
See the model_card.md

---

## Limitations and Risks

- It only works on a tiny catalog
- It does not understand lyrics or language
- It leans more on genre and mood. 

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)


