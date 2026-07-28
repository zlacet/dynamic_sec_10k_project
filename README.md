# Dynamic SEC 10-K RAG Agent

A retrieval-augmented generation (RAG) system that answers natural-language questions about any publicly traded company's most recent SEC 10-K annual report — not a fixed list of companies. Ask about a company by name, and the system detects the ticker, fetches and indexes its latest 10-K directly from SEC EDGAR on the fly, and answers grounded strictly in that filing's own text.

## Why this project

Most RAG demos are built around a fixed, pre-indexed set of documents. This one removes that constraint: mention any publicly traded company by name in a question, and the system detects the ticker, fetches its current 10-K directly from SEC EDGAR, indexes it on the spot, and answers grounded strictly in that filing's own text. Every later question about that company is served from the vector store like normal, and the system re-checks EDGAR each time to make sure it's never answering from a stale filing.

## How it works

Everything lives in a single notebook, `agent.ipynb`, built around the `edgartools` library for EDGAR access:

1. **Ticker validation** — a lookup table built from SEC's public `company_tickers.json` confirms a ticker is real before anything else runs.
2. **Fetching the latest 10-K** — `edgartools`' `Company(ticker).get_filings(form="10-K")` returns a company's 10-K filings; the system takes the first (most recent) one and excludes 10-K/A amendments.
3. **Section-aware parsing and chunking** — `edgartools` exposes a filing's parsed items (`Item 1`, `Item 1A`, `Item 7`, etc.) directly; long sections are further split into ~6,000-character pieces so retrieved context stays a manageable size.
4. **Embedding and indexing** — chunks are embedded with `text-embedding-3-small` and stored in a persistent ChromaDB collection, tagged with ticker, section, and the filing's SEC accession number.
5. **Staleness check** — every time a ticker is queried, the system re-fetches the latest filing's accession number and compares it to what's indexed. If a newer 10-K has been filed since the ticker was last indexed, the old chunks are deleted and replaced automatically, so answers never come from an outdated filing.
6. **Ticker auto-detection** — if a question doesn't specify a company explicitly, an LLM call (`extract_tickers`) pulls the likely ticker(s) out of the question text, validated against the real ticker list before use.
7. **Retrieval + generation** — `retrieve()` does a ticker-scoped similarity search against the vector store; `ask()` builds a prompt instructing the model to answer only from retrieved context, citing the ticker, and to say so rather than guess if the context doesn't contain the answer. Generation uses `temperature=0` for reproducibility, with exponential backoff retry on rate limits.

## Setup

**1. Prerequisites.** You'll need Python 3.10 or later and `pip` installed on your machine. You'll also need an OpenAI API key — sign up or log in at [platform.openai.com](https://platform.openai.com), go to API keys, and create one. Note that using the API incurs a small cost per request; this project uses cheap models (`gpt-4o-mini`, `text-embedding-3-small`), so cost per question is minimal, but it isn't free.

**2. Get the code.** Clone or download this repository, then open a terminal in the project folder.

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Add your API key.** Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace the placeholder with your real OpenAI API key, so it looks like:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

**5. Launch Jupyter and open the notebook:**

```bash
jupyter notebook
```

This opens a browser tab showing the project folder. Click `agent.ipynb` to open it.

**6. Run the notebook.** Run the cells from top to bottom (Shift+Enter on each, or use "Run All" from the toolbar). The first cells set up the environment and define the functions; nothing happens until you get to a cell that actually calls `ask(...)`.

**7. Ask a question.** The last several cells contain example questions you can run as-is, or edit to ask about any public company. The first question about a given company will take a bit longer (fetching and indexing its 10-K from EDGAR); every later question about that same company answers almost instantly from the local vector store, unless a newer 10-K has since been filed, in which case it re-indexes automatically.

## Asking a question

```python
# Just ask a question — ticker auto-detected from the question text
answer, sources = ask("What does Nike say about the risk of counterfeit products harming its brand?")

# Specify the ticker explicitly — skips auto-detection entirely
answer, sources = ask("What labor-related risk does Costco disclose?", relevant_tickers=["COST"])

# Override how many chunks are retrieved per company (default is 20)
answer, sources = ask("What does Tesla say about its battery supply chain?", n_per_ticker=30)
```

## Known limitations

**Retrieval ranking can miss real, relevant content.** Several test questions exposed cases where a specific risk factor exists verbatim in a 10-K, but other sections sharing surface vocabulary with the question outrank it in embedding similarity, so the correct chunk never makes it into context. Increasing `n_per_ticker` (chunks retrieved per company, currently 20) helps in some cases but not all — a question about Nike's counterfeit-goods risk still failed to surface the relevant paragraph even after doubling retrieval depth. In every observed case, the system fails safe (declining to answer) rather than hallucinating, but it can under-retrieve content that's genuinely present in the source document.

**Ticker auto-detection isn't perfectly reliable.** `extract_tickers` asks an LLM to pull a ticker symbol out of a question's plain-English company reference. This mostly works, but it isn't guaranteed to succeed on every phrasing, and there's no fallback beyond `validate_tickers` filtering out obviously invalid output. The workaround is to pass `relevant_tickers` explicitly when auto-detection fails.

## Data source

Filings are fetched live from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) via `edgartools`, a public source. No non-public or licensed data is used.

## Tech stack

Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`), ChromaDB, `edgartools`, pandas, Jupyter.
