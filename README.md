# Dynamic SEC 10-K RAG Agent

This agent answers natural-language questions about any publicly traded company's most recent SEC 10-K annual report. Ask about any company by name, and the system detects the ticker, fetches and indexes its latest 10-K directly from SEC EDGAR on the fly, then answers using only content it finds in that filing.

## Why this project

Most RAG demos are built around a fixed, pre-indexed set of documents chosen ahead of time. This one removes that constraint entirely: any publicly traded company works, because filings are fetched and indexed at query time rather than pre-loaded. The system also re-checks EDGAR on every query, so a company that was indexed months ago never gets stale answers if a newer 10-K has since been filed.

## How it works

Everything lives in a single notebook, `agent.ipynb`, built around the `edgartools` library for EDGAR access:

1. **Ticker validation**: a lookup table built from SEC's public `company_tickers.json` confirms a ticker is real before anything else runs.
2. **Fetching the latest 10-K**: `edgartools`' `Company(ticker).get_filings(form="10-K")` returns a company's 10-K filings; the system takes the first (most recent) one and excludes 10-K/A amendments.
3. **Section-aware parsing and chunking**: `edgartools` exposes a filing's parsed items (`Item 1`, `Item 1A`, `Item 7`, etc.) directly; long sections are further split into ~6,000-character pieces so retrieved context stays a manageable size.
4. **Embedding and indexing**: chunks are embedded with `text-embedding-3-small` and stored in a persistent ChromaDB collection, tagged with ticker, section, and the filing's SEC accession number.
5. **Staleness check**: every time a ticker is queried, the system re-fetches the latest filing's accession number and compares it to what's indexed. If a newer 10-K has been filed since the ticker was last indexed, the old chunks are deleted and replaced automatically, so answers never come from an outdated filing.
6. **Ticker auto-detection**: if a question doesn't specify a company explicitly, an LLM call (`extract_tickers`) pulls the likely ticker(s) out of the question text, validated against the real ticker list before use.
7. **Retrieval + generation**: `retrieve()` does a ticker-scoped similarity search against the vector store; `ask()` builds a prompt instructing the model to answer only from retrieved context and to say so rather than guess if the context doesn't contain the answer. Generation uses `temperature=0` for reproducibility, with exponential backoff retry on rate limits.

## Setup

**1. Prerequisites.** You'll need Python 3.10 or later and `pip` installed on your machine. You'll also need an OpenAI API key (sign up or log in at [platform.openai.com](https://platform.openai.com), go to API keys, and create one). Note that using the API incurs a small cost per request; this project uses cheap models (`gpt-4o-mini`, `text-embedding-3-small`), so cost per question is minimal, but it isn't free.

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

**7. Ask a question.** The last several cells contain example questions you can run as-is, or edit however you like: change the question, the company, or both. Any question about any public company works. The first question about a given company will take a bit longer (fetching and indexing its 10-K from EDGAR); every later question about that same company answers almost instantly from the local vector store, unless a newer 10-K has since been filed, in which case it re-indexes automatically.

## Asking a question

```python
# Just ask a question: ticker auto-detected from the question text
answer, sources = ask("What does Nike say about the risk of counterfeit products harming its brand?")

# Specify the ticker explicitly (skips auto-detection entirely)
answer, sources = ask("What labor-related risk does Costco disclose?", relevant_tickers=["COST"])

# Override how many chunks are retrieved per company (default is 20)
answer, sources = ask("What does Tesla say about its battery supply chain?", n_per_ticker=30)
```

## Results and Validation

To confirm the system produces accurate, non-hallucinated answers, six demonstration questions, one per company, were run through the notebook and manually cross-checked against each company's actual, full-text SEC 10-K filing.

### Tesla

**Question:** What legal proceedings or litigation does Tesla disclose in its 10-K?

**Answer:** The system listed 11 distinct proceedings: the Delaware Chancery and federal derivative suits over the 2018 going-private episode, the 2022 SEC-settlement fiduciary-duty suit, three 2024 derivative actions tied to Musk/X Corp/xAI, the CRD and EEOC discrimination proceedings, the Benavides Autopilot verdict ($129M compensatory / $200M punitive), and three Autopilot/FSD/Robotaxi-related class actions.

**Validation:** Every date and dollar figure matched Tesla's actual 10-K exactly. No discrepancies found.

### Amazon

**Question:** Who does Amazon identify as its main competitors?

**Answer:** The system listed roughly 10 competitor categories (physical/e-commerce/omnichannel retailers, media companies, search/social platforms, e-commerce service providers, fulfillment/logistics companies, IT/cloud/AI companies, consumer electronics makers, grocery sellers, advertisers, and healthcare providers) plus three bullets on competitive factors.

**Validation:** Matched the filing's "Competition" section one-to-one, category for category.

### Costco

**Question:** What does Costco disclose about risks related to membership fee renewal rates?

**Answer:** The system covered the importance of membership loyalty and growth, the link between renewal rates and profitability, the brand/reputation risk to renewals, and the specific renewal figures (92.3% U.S./Canada, 89.8% worldwide, FY2025).

**Validation:** All claims and figures matched the filing exactly.

### Disney

**Question:** How does Disney break down its business into reportable segments?

**Answer:** The system correctly identified the three segments (Entertainment, Sports, Experiences) and their lines of business, including Linear Networks, Direct-to-Consumer (Disney+/Hulu), Content Sales/Licensing (with the Tata Play stake and National Geographic magazine), ESPN, and the Experiences properties (Disney Cruise Line, Disney Vacation Club, international parks).

**Validation:** Matched Item 1's business description exactly.

### NVIDIA

**Question:** What does NVIDIA disclose about its employee compensation structure?

**Answer:** The system covered stock-based compensation figures ($14.8B unearned SBC, $6.4B total FY26 SBC split between R&D and SG&A), the 2007 Equity Incentive Plan's share counts, ESPP terms, and workforce stats (80%+ technical roles, 3.7% turnover).

**Validation:** All figures matched Note 3 and the Human Capital Management section exactly.

### Nike

**Question:** What does Nike say about the risk of counterfeit products harming its brand?

**Answer:** The system covered the periodic discovery of counterfeits, the sales/brand risk from failed enforcement, the risk of shifting consumer preference, and the expense/liability risk from IP protection efforts.

**Validation:** Matched the Item 1A risk factor almost verbatim.

**Overall outcome:** across all 6 companies and roughly 45 individual factual claims, no fabricated facts, misattributed figures, or hallucinated numbers were found. Every claim traced back to a real, locatable statement in the source filing. This validation was performed after refining the generation prompt earlier in the project to improve enumeration completeness and answer quality, confirming the changes introduced no factual regressions.

## Known limitations

**Retrieval depth affects whether the correct content is surfaced.** At the default `n_per_ticker` (20), some questions fail even though the relevant content exists verbatim in the filing: the correct chunk doesn't make it into the top-N results, likely because other sections share enough surface vocabulary to outrank it in embedding similarity. Increasing `n_per_ticker` (e.g., to 30) resolves this in observed cases, but the system doesn't automatically detect when a question needs deeper retrieval, so getting a complete answer can depend on manually tuning that parameter per question. In every observed case, the system fails safe (declining to answer) rather than hallucinating.

**Ticker auto-detection isn't perfectly reliable.** `extract_tickers` asks an LLM to pull a ticker symbol out of a question's plain-English company reference. This mostly works, but it isn't guaranteed to succeed on every phrasing, and there's no fallback beyond `validate_tickers` filtering out obviously invalid output. The workaround is to pass `relevant_tickers` explicitly when auto-detection fails.

## Data source

Filings are fetched live from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) via `edgartools`, a public source. No non-public or licensed data is used.

## Tech stack

Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`), ChromaDB, `edgartools`, pandas, Jupyter.
