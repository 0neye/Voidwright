# Contribution Guidelines

This is a large, fast-moving project. Using coding agents is highly recommended.
This will be covered in greater detail later.

## Setup

Install dev dependencies after cloning:

```bash
pip install -e .[dev]
```

Then copy the ship files you want to use into a new folder in your project directory. You'll need some ships for the program to work on. Recommended ships are your own library and the built-in library excluding stations.

See `README.md` in the project root for further information.

**For Agents:**
Read `CLAUDE.md` (or `AGENTS.md` for non-Claude agents) for a full architecture overview and important conventions before making changes.

## First Contributions

A good starting place for first contributions are things like extra validation checks for parts during generation, new nodes during graph expansion (once implemented), and fixing bugs.

If you aren't sure what to work on but want to contribute, feel free to ask me (0neye) in the Excelsior project thread.

## Testing

Agent instructions already include testing guidance, but you can also see it in `tests/README.md`.
As a human, it's your job to extend this beyond automated testing and actually hunt for issues in-game. Validate your changes are doing what you intend.

## AI Tools

The recommended models and harnesses are:
1. Codex with GPT-5.4 or GPT-5.3-Codex
2. Claude Code with Opus or Sonnet 4.6
2. Cursor with any of the above models

For students, you can currently get a free year of Cursor Pro, which is a fantastic deal, and the recommended low-risk way to get started.
A $20/month ChatGPT or Claude subscription give you decent usage as well.

### Additional AI Tools

Cognition provides a couple of useful free services.

The first is DeepWiki, which you can use by replacing `github` in the URL of this repo page with `deepwiki`. Use this to understand the codebase and ask questions to the QnA agent.

The second is Devin Review, which can be used to review open pull requests by replacing `github` in the URL with `devinreview`. **This should be used for all opened pull requests.**

## Commit Messages

Every commit must include the AI model used (if any) in the footer:

```
<subject line>

<body if needed>

Co-Authored-By: <model-name>
```

If no AI was used, omit the footer.

## Pull Requests

If you make a contribution and open a pull request, you must use Devin Review (see above), and fix any valid issues it raises. Re-verify on your own machine in-game or with custom tests to make sure your contribution works. Once you're reasonably confident, please ping me on Discord.

> **Note for maintainers:** `CLAUDE.md` and `AGENTS.md` must always be kept in sync. If you update one, update the other.