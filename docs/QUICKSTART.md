# Quickstart

From nothing to a grounded quote in about a minute, then the parts worth knowing
before you trust the answers.

## 1. Prerequisites

- Python 3.10 or newer.
- `poppler-utils`, which supplies the `pdftotext` binary.

```bash
# Debian / Ubuntu
sudo apt-get install poppler-utils
# Fedora
sudo dnf install poppler-utils
# macOS
brew install poppler
```

Check it:

```bash
pdftotext -v
```

That is the whole external dependency list. There is no Docker image, no model
download and no GPU requirement, because the server never calls an LLM.

## 2. Install

```bash
cd science2code-mcp
pip install -e .
```

That gives you the library and the `science2code` command line, both of which
run on the standard library alone.

The MCP server needs one wheel, `fastmcp`, and it is an extra rather than a
dependency so that the anchoring code stays readable on a bare Python. Install
it if you want to register the server with an agent, which is most people:

```bash
pip install -e ".[mcp]"
```

To work on the project, install the development extra, which is `pytest` and
`ruff`:

```bash
pip install -e ".[dev]"
```

The extras combine: `pip install -e ".[mcp,dev]"`.

## 3. Build a corpus

A corpus is a directory of PDFs plus a `manifest.json` that the tool writes for
you. That is all it is. It is not a database, it is not a reference manager
export, and nothing outside the directory has to exist.

```bash
mkdir -p ~/papers
cp ~/Downloads/*.pdf ~/papers/
science2code index ~/papers
```

`index` runs `pdftotext` in reading order over every PDF, writes a `.txt` beside
each one, and writes `manifest.json` in the directory. Indexing an unchanged
corpus twice produces an identical manifest: the file carries no timestamps, on
purpose, so that a diff means something.

Then:

```bash
science2code status ~/papers
```

`status` tells you what is held, what has gone stale because the extractor or
the normaliser changed, and which PDFs have no usable text layer.

Exit codes: `0` clean, `1` finished with problems worth your attention, `2` the
command could not run at all.

### Paper ids

The id is the filename stem, with characters that would make a path or a JSON
key awkward folded to an underscore. There is no `REF-nn` scheme and no
renumbering, because a renumbering scheme means every stored reference to a
paper can go stale for a reason that has nothing to do with the paper.

So `nygard-2011-documenting-architecture-decisions.pdf` becomes the id
`nygard-2011-documenting-architecture-decisions`. If you want short ids, rename
the files before you index. Ids appear in every tool call and in every result.

### Correcting metadata

The manifest is the source of metadata, and it is a plain JSON file you can open
and fix. Extraction takes its header fields from the manifest and never from the
PDF, so a corrected title flows straight into the next extraction.

Each entry looks like this:

```json
{
  "paper_id": "nygard-2011-documenting-architecture-decisions",
  "title": "Documenting Architecture Decisions",
  "authors": ["Michael Nygard"],
  "year": 2011,
  "venue": null,
  "doi": null,
  "pdf_path": "nygard-2011-documenting-architecture-decisions.pdf",
  "text_path": "nygard-2011-documenting-architecture-decisions.txt",
  "pdf_sha256": "...",
  "text_sha256": "...",
  "pages": 3,
  "held": true
}
```

A field you did not fill in stays `null`. Nothing is guessed from the PDF, so a
paper with no recorded year comes back with no year rather than with an invented
one. `held: false` means the paper is known but no usable text is available for
it.

Manifest validation fails closed. A bad field raises an error naming that field,
and no partial corpus is loaded: if you get a corpus object, you get a whole one.

### The invariant, where it bites

**This tool never deletes a file.** A PDF sitting in the directory with no
manifest entry is reported, never removed. A `.txt` file that differs from a
fresh extraction is left alone and reported, and `index` will only replace it if
you pass `--force`. Where a file ought to go away, the command to remove it is
printed for you to run.

## 4. Register the server

The server is `python -m science2code.server`, and it takes the corpus
directory from the environment variable `SCIENCE2CODE_CORPUS` rather than from
an argument. It needs the `mcp` extra installed. There is no `science2code
serve` subcommand: `science2code --help` lists everything the command line can
do, and it is `index` and `status`.

In Claude Code:

```bash
claude mcp add science2code \
  --env SCIENCE2CODE_CORPUS=/home/you/papers \
  -- python -m science2code.server
```

Any client that takes a JSON server block:

```json
{
  "mcpServers": {
    "science2code": {
      "command": "python",
      "args": ["-m", "science2code.server"],
      "env": { "SCIENCE2CODE_CORPUS": "/home/you/papers" }
    }
  }
}
```

Use the `python` from the environment you installed into, by absolute path if
your client does not inherit your shell. With no `SCIENCE2CODE_CORPUS` set, the
server looks for a directory called `corpus` relative to wherever it was
started, which is almost never what you want. Every tool call then comes back
`CORPUS_UNAVAILABLE`, and the `not_available` entry carries the path it tried
in its `detail` field, which is usually enough to see what went wrong.

## 5. First questions to ask it

Ask your agent to verify something you already know is in a paper:

> Use verify_quote to check whether "ADRs will be numbered sequentially and
> monotonically." appears in nygard-2011-documenting-architecture-decisions.

You should get `VERBATIM_EXACT` back, with a page and a character interval.

Now ask it to verify something that is not there:

> Use verify_quote to check whether "ADRs should be rewritten whenever the
> architecture changes." appears in the same paper.

You should get `NOT_LOCATABLE`. That is the tool working, not failing. A refusal
is a valid answer, and it is the one that makes every other answer worth
something.

Then ask what a paper actually says:

> Use find_passage on that paper for "mark it as superseded".

You get passages in the document's own words, with locators. It is a lexical
search over the extracted text: it does not rank by meaning, does not summarise,
and reports `hit_count: 0` rather than handing back the nearest thing it found.

Ask it for the single word "superseded" instead and you get `NOT_LOCATABLE`
with the reason: a query under 12 characters occurs in too many places for one
location to be the answer, so no search is run and the tool asks for more of
the sentence. Both tools apply that floor, and it is measured in characters
after normalisation, not in words.

## 6. Reading a result

Every response carries exactly one `outcome` from a closed vocabulary.

| Outcome | What happened | Safe to quote as the document's words? |
|---|---|---|
| `VERBATIM_EXACT` | your string is literally in the document | yes |
| `VERBATIM_RELAXED_EXTRACTOR_DAMAGE` | it is literally there once an intra-word hyphen is removed and case is folded, which is extractor damage rather than quote damage | yes |
| `PASSAGE_RELOCATED_QUOTE_DIFFERS` | a passage was found, and its text differs from yours | **no** |
| `NOT_LOCATABLE` | nothing close enough exists in that document | no |
| `SOURCE_NOT_HELD` | the paper is known but no usable text is held for it | no |
| `SOURCE_UNKNOWN` | no paper with that id is held locally | no |
| `CORPUS_UNAVAILABLE` | the corpus could not be read at all | no |
| `OK` | a `find_passage` query ran to completion. The hits inside it carry their own outcomes | no |

On `PASSAGE_RELOCATED_QUOTE_DIFFERS` the response carries three deliberately
named fields: `document_text` says whose words those are, `your_text` says whose
words yours were, and `char_diff` is the distance between them. `char_diff` is a
short listing, one entry per difference, and it looks like this:

```
char diff: - quote, + document
  @50 after 'entially and monotonical'
    + 'l'
  @51 after 'ntially and monotonicaly'
    - '.'
```

The number after `@` is the offset in `your_text` where the difference starts,
and the quoted string after `after` is the text just before it. A line starting
with a minus sign holds characters your text has and the document does not; a
line starting with a plus sign holds characters the document has and yours does
not. The response carries that explanation with it, in `char_diff_legend`, so
you never have to remember which way round the signs go.

The correct move on that outcome is to replace your text with the document's,
not to argue with the score. A caller that treats a relocated passage as a hit
is defeating the entire point of the tool.

**Read the outcome, never the score.** The score exists so you can see how close
something got, not so you can set your own threshold. There is no threshold on
the two character-identity outcomes, and reintroducing one in your own code
throws away the property that makes them trustworthy.

## 7. When extraction finds nothing

Some PDFs have no text layer. An image-only scan yields a handful of characters
per page, or none. Those papers are recorded with `held: false`, and every
`verify_quote` against them returns `SOURCE_NOT_HELD` rather than pretending
your quote was wrong. `index --floor N` sets the characters-per-page threshold
below which a PDF is treated this way.

There is no OCR in this tool. Your options are to obtain a text-bearing copy, or
to quote the paper by hand and accept that this tool cannot check you. That is
an honest boundary and it is stated rather than engineered around.

## 8. Why your quote might miss

In rough order of how often it happens:

1. **You paraphrased.** The commonest cause. `PASSAGE_RELOCATED_QUOTE_DIFFERS`
   will hand you the real wording.
2. **A ligature or a curly quote.** Normalisation handles the usual cases, but a
   quote copied from a rendered web page can differ from the PDF's own bytes.
3. **A line break inside a hyphenated word.** This is what
   `VERBATIM_RELAXED_EXTRACTOR_DAMAGE` exists for, and it still counts as
   character identity.
4. **The passage crosses a page boundary or a figure caption.** Extraction is
   linear text, and a caption sitting mid-column can interrupt a sentence.
5. **The quote is from a different paper.** This is the case the tool exists to
   catch. The score will be low and the outcome will be `NOT_LOCATABLE`.

## 9. One thing the tool will not do for you

It will not tell you whether the passage is evidence for your claim, and it will
not tell you which relevant work you never read. Neither is mechanically
decidable. The server says so in every response it sends, and the reasoning is
in the README's ceiling section.
