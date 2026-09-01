# science2code-mcp

A local MCP server that grounds coding agents in scientific papers you already
hold. An agent checks a quote against your PDFs and gets back the paper's own
words with a page and a locator, or a typed refusal. No language model is in the
loop, so nothing can hallucinate a passage, and it never claims whose idea a
sentence is.

## What it is for

You are building software and want its design grounded in real literature. Drop
the papers in a folder; when an agent cites one ("Paper X says Y"), this server
confirms Y is verbatim in X, or refuses. It keeps citations honest.

It does not summarise, rank, or decide which paper is relevant. Those remain
your work and the agent's. This is the fact-checker underneath them.

## What it is not, and the context-window question

Be clear about this before you rely on it. An agent can read the extracted
`.txt` files, and doing so puts them in the context window, which is the cost
you are probably trying to avoid. **science2code does not remove that cost.** It
was never a "read all my papers cheaply" tool. It removes a different cost: to
verify a quote, the agent sends one string and gets a small answer back, without
loading the paper. So it makes checking cheap and trustworthy; it does not make
comprehension free. If what you want is "give the agent the relevant bits
without dumping everything in," that is semantic retrieval, a different tool this
is not.

## Install

Requires Python 3.10 or newer and `poppler-utils` (which supplies `pdftotext`).

```bash
pip install "science2code-mcp[mcp]"     # from a checkout: pip install -e ".[mcp]"
```

The core has zero third-party dependencies and is stdlib only. The `mcp` extra
adds the server runtime (`fastmcp`); `dev` adds pytest and ruff.

## Quickstart

```bash
mkdir ~/papers && cp *.pdf ~/papers/
science2code index ~/papers      # extract text once, write manifest.json
science2code status ~/papers     # what is held, stale, or has no text layer
```

Then register the server with your MCP client and point the environment variable
`SCIENCE2CODE_CORPUS` at `~/papers`. The full walkthrough, including correcting a
title and handling a scanned PDF, is in [docs/QUICKSTART.md](docs/QUICKSTART.md).

## The two tools an agent sees

- **`verify_quote(text, paper_id?)`**: is this quote really in this paper?
  Returns the document's own characters with a page and a character locator, or a
  typed refusal. Omit `paper_id` to check every held document.
- **`find_passage(query, paper_ids?)`**: where does this phrase occur? A literal
  search that returns passages in the paper's words, or zero hits. It is not
  semantic, does not rank by relevance, and does not summarise.

## Outcomes

Every response carries exactly one `outcome` from a closed set, so a caller can
enumerate every answer the server can give.

| outcome | meaning | asserts identity? |
|---|---|---|
| `VERBATIM_EXACT` | the string occurs literally in the document | yes |
| `VERBATIM_RELAXED_EXTRACTOR_DAMAGE` | it occurs once an intra-word hyphen and case, the signature of extractor damage, are folded | yes |
| `PASSAGE_RELOCATED_QUOTE_DIFFERS` | a passage was located and your string differs; carries a character diff | no |
| `NOT_LOCATABLE` | nothing above threshold; a result, not an error | no |
| `SOURCE_NOT_HELD` | the document is known but no usable text is held (a scan, say) | no |
| `SOURCE_UNKNOWN` | no document with that identifier is held | no |
| `CORPUS_UNAVAILABLE` | the local corpus could not be read | no |
| `OK` | a `find_passage` query ran to completion | no |

The two identity outcomes use pure character comparison with no similarity
threshold, so a paraphrase can never be returned as the document's own words.

## The ceiling

It reports whether a string is in a document and where. It does **not** judge
whether a passage supports a claim, and it cannot detect a relevant work you
never read. Both are human judgements, and it never asserts whose claim a
sentence carries. Every response says so.

## It never deletes a file

No deletion primitive is imported anywhere in the package, a test over the syntax
tree keeps it that way, and writes go through one temp-and-replace helper. Where
something ought to be removed, the command is printed for you to run.

## Optional: build the corpus from Zotero (read only)

You never need Zotero. But if your papers live there, a separate, strictly
read-only command can fill the folder from a tagged subset over Zotero's built-in
local API:

```bash
science2code-zotero sync --tag mytag --out ./papers   # symlinks the tagged PDFs
science2code index ./papers
```

It only reads Zotero (HTTP GET), never writes to it, and removes only links it
made itself. `science2code-zotero annotations --tag mytag` lists your highlights
as candidate quotes. The core never imports any of this.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE). To cite the software, see
[CITATION.cff](CITATION.cff).
