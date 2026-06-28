# Redrob AI Candidate Ranker

> Ranks candidates the way a great recruiter would — by understanding context, not just matching keywords.

## The Problem

Keyword-based ATS filters eliminate great candidates because they can't see:
- Career trajectory and potential
- Behavioral signals and initiative
- The difference between "knows Python" and "ships ML systems in Python"
- Platform engagement that predicts hire success

## Our Solution

A **hybrid AI ranking pipeline** combining rule-based scoring with LLM holistic assessment.

```
Job Description  ──►  LLM JD Parser  ──►  Structured Requirements
                                                    │
Candidate Profiles ──► Multi-dim Scorer ──────────►│
                   ──► LLM Holistic Rater ─────────►│
                                                    ▼
                                           Ranked Shortlist + Explanations
```

## Architecture

### 5 Scoring Dimensions

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Skill Fit | 30% | Semantic skill matching (handles synonyms, related tech) |
| Experience | 20% | Fit to seniority range (penalizes over/under-qualification) |
| Potential | 15% | GitHub activity, open source, certifications, impact |
| Activity | 10% | Platform engagement, response rate, recency |
| Culture | 10% | Career summary signals vs. JD culture requirements |
| LLM Holistic | 15% | Claude's full-picture assessment with recruiter reasoning |

### Key Design Decisions

1. **LLM for JD parsing** — extracts "what this role really needs" not just keywords
2. **Semantic skill matching** — understands PyTorch ≈ Torch, K8s ≈ Kubernetes
3. **Overqualification detection** — Priya > 8 yrs for a 4-8 yr role gets flagged
4. **Activity signals** — a candidate active yesterday beats one last seen 3 weeks ago
5. **LLM holistic re-scoring** — final layer that weighs the full picture

## Setup

```bash
git clone https://github.com/yourusername/redrob-ranker
cd redrob-ranker
pip install -r requirements.txt

# Run with full AI (requires Anthropic API key in env)
export ANTHROPIC_API_KEY=your_key_here
python3 main.py

# Run with your own data
python3 main.py --candidates path/to/candidates.csv --jd path/to/jd.txt

# Run rule-based only (no API key needed)
python3 main.py --no-llm
```

## Input Format

`candidates.csv` expects these columns:

| Column | Type | Description |
|--------|------|-------------|
| id | str | Unique candidate ID |
| name | str | Full name |
| title | str | Current job title |
| years_exp | int | Total years of experience |
| skills | list/str | Comma-separated skills |
| education | str | Highest degree |
| career_summary | str | Profile summary |
| achievements | str | Key achievement(s) |
| github_repos | int | Number of public repos |
| certifications | list/str | Relevant certifications |
| open_source_contributions | bool | True/False |
| platform_activity_score | int | 0-100 platform score |
| response_rate | int | 0-100 response rate |
| last_active_days | int | Days since last active |

## Output Format

`output/ranked_candidates.csv`:

| Column | Description |
|--------|-------------|
| Rank | 1 = best fit |
| Total Score (0-100) | Composite score |
| Recommendation | Strong Yes / Yes / Maybe / No |
| Skill/Experience/Potential/Activity/Culture/LLM Scores | Dimension breakdown |
| Strengths | Top 2-3 specific strengths for this role |
| Gaps | Honest gaps or risks |
| Recruiter Note | LLM-generated recruiter assessment |

## Project Structure

```
redrob-ranker/
├── main.py                  # Entry point
├── src/
│   └── ranker.py            # Core ranking engine
├── data/
│   ├── generate_data.py     # Synthetic data generator
│   └── candidates.csv       # Generated/real candidate data
├── output/
│   └── ranked_candidates.csv # Submission output
├── requirements.txt
└── README.md
```

## Results (Demo — ML Engineer Role)

| Rank | Candidate | Score | Recommendation |
|------|-----------|-------|---------------|
| 1 | Priya Sharma | 79.4 | Strong Yes |
| 2 | Amit Joshi | 71.7 | Yes |
| 3 | Pooja Reddy | 67.6 | Yes |
| 4 | Sneha Patel | 66.2 | Maybe |
| 5 | Arjun Mehta | 61.8 | Maybe |

## Built for the Redrob Challenge

This solution was built for the Redrob AI Hiring Challenge. The goal: make hiring smarter by ranking candidates the way a great recruiter would — understanding context, trajectory, and genuine fit rather than keyword overlap.
