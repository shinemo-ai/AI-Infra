# AI-Infra

A knowledge base for our work on open-source LLM inference frameworks and related infrastructure.

We run models in production using projects such as [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and others. This repository collects two kinds of notes:

1. **Bugfixes** — Issues found in production or staging, along with patches, branches, and context that may not yet be reviewed or merged upstream.
2. **Research notes** — Findings from performance tuning, architecture exploration, and experiments that are worth keeping for the team.

Upstream communities move quickly and review timelines vary. Documenting fixes and learnings here helps us deploy safely, onboard teammates, and contribute back when ready.

---

## Repository Layout

```
AI-Infra/
├── sglang/
│   └── bugfix/          # SGLang production bugfixes and patch notes
├── vllm/                # (planned) vLLM bugfixes and patch notes
└── research/            # (planned) shared research notes and write-ups
```

Each framework directory may grow its own structure (e.g. `bugfix/`, `notes/`) as needed.

---

## Contributing

When adding material:

- **Bugfixes** — Record symptoms, root cause, branch/commit, related code changes, and upstream PR status if applicable. See [sglang/bugfix/README.md](sglang/bugfix/README.md) for an example.
- **Research** — Prefer concise write-ups with motivation, approach, results, and open questions.

Keep entries in English unless a specific doc intentionally targets another audience.
