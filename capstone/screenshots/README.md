# Screenshots

Capture these from the running app for the submission:

| File | What to capture |
|---|---|
| `01_app_overview.png` | Landing page, sidebar showing the active generation backend and corpus table |
| `02_answer_with_citations.png` | Answer tab with inline citations and the resolved source list |
| `03_evidence_by_branch.png` | A branch tab with one passage expanded, showing fused score, rerank score and retrieval path |
| `04_routing.png` | Routing tab for "Should we launch a budget tablet? ..." showing per-branch scores against the cutoff |
| `05_diagnostics.png` | Diagnostics tab: citation grounding, review sentiment, stage timings |

```bash
cd capstone
streamlit run app.py
```

Then open <http://localhost:8501> and work through the four tabs.
