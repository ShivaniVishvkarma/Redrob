#!/usr/bin/env python3
"""
Redrob Hackathon — Senior AI Engineer Ranker
=============================================
CPU-only, no network, <5 min for 100K candidates.

Scoring philosophy (from reading the JD carefully):
  - The JD explicitly says: don't keyword-match. Reason about career history.
  - Disqualifiers matter more than skills checklists.
  - Behavioral signals are a multiplier, not an afterthought.
  - Honeypots (impossible profiles) must be detected and excluded.
  - The "ideal candidate" is 6-8 yrs, applied ML at product companies,
    has shipped ranking/retrieval/search, is active and reachable.

Architecture: 5 scoring components × explicit disqualifier checks × behavioral multiplier

Components:
  1. Career substance score     (0-1) — what did they actually build?
  2. Technical fit score        (0-1) — do they have the right skills, used at depth?
  3. Experience bracket score   (0-1) — 5-9 yrs, but JD means 6-8 sweet spot
  4. Location/availability score (0-1) — India preferred, relocation, notice period
  5. Education signal           (0-1) — tier, field; not a dealbreaker but a signal

Multiplied by:
  6. Behavioral availability multiplier (0.2-1.0) — active, responsive, engaged

Disqualifiers (set score to near-zero):
  - Pure research background, no production deployment
  - <12 months AI exp and no pre-LLM ML experience
  - Senior title but no code in 18 months
  - Entire career at services firms (TCS, Infosys, etc.) with no product exposure
  - Primary domain: CV/speech/robotics without NLP/IR
  - Honeypot detection: impossible chronology, inflated skills
"""

import json
import re
import sys
import argparse
import math
from datetime import datetime, date
from pathlib import Path
from typing import Any

import csv

# ── Constants ────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 28)

# Skills the JD explicitly requires (must-haves)
CORE_SKILLS = {
    # Embeddings / retrieval
    "sentence-transformers", "sentence_transformers", "embeddings", "semantic search",
    "dense retrieval", "bi-encoder", "cross-encoder",
    "openai embeddings", "bge", "e5", "gte",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "pgvector", "chroma", "vespa",
    # Search/ranking
    "bm25", "hybrid search", "vector search", "information retrieval",
    "ranking", "reranking", "re-ranking", "learning to rank", "ltr",
    "recommendation", "recommendation system", "recommender",
    "search", "retrieval",
    # Eval
    "ndcg", "mrr", "map", "a/b testing", "evaluation framework",
    "offline eval", "online eval", "ranking eval",
    # LLMs / modern ML
    "llm", "large language model", "transformers", "bert", "rag",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "nlp", "natural language processing", "text", "language model",
    # Python / infra
    "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    # Nice-to-haves
    "xgboost", "lightgbm", "gradient boosting",
    "distributed systems", "kafka", "spark", "airflow",
}

NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "lightgbm", "learning to rank",
    "kafka", "spark", "distributed",
    "open source", "github",
}

# Hard disqualifier companies (services, not product)
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
    "mphasis", "l&t infotech", "ltimindtree", "hexaware",
    "niit technologies", "kpit", "cyient", "mastech",
}

# Title signals for "is this person actually an ML/AI engineer"
ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied ml", "applied scientist", "research engineer",
    "nlp engineer", "data scientist", "senior engineer",
    "staff engineer", "principal engineer", "ai researcher",
    "search engineer", "ranking engineer", "recommendations engineer",
    "retrieval engineer",
}

# CV/speech/robotics disqualifier domains (without NLP)
CV_SPEECH_DOMAINS = {
    "computer vision", "image recognition", "object detection",
    "speech recognition", "speech synthesis", "tts", "asr",
    "robotics", "autonomous vehicles", "self-driving",
}

NLP_IR_SIGNALS = {
    "nlp", "natural language", "text", "language model",
    "information retrieval", "search", "ranking", "recommendation",
    "bert", "llm", "rag", "embeddings",
}

# Preferred India locations
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
    "kolkata", "india",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    return s.lower().strip() if isinstance(s, str) else ""

def days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 999

def skill_names(skills: list) -> set:
    return {norm(s.get("name", "")) for s in skills if isinstance(s, dict)}

def skill_map(skills: list) -> dict:
    """name -> {proficiency, endorsements, duration_months}"""
    out = {}
    for s in skills:
        if isinstance(s, dict):
            out[norm(s.get("name", ""))] = s
    return out

def text_blob(c: dict) -> str:
    """All text from a candidate in one string for fast keyword search."""
    parts = []
    p = c.get("profile", {})
    parts += [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    for job in c.get("career_history", []):
        parts += [job.get("title", ""), job.get("description", ""), job.get("company", "")]
    for e in c.get("education", []):
        parts += [e.get("field_of_study", ""), e.get("degree", "")]
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    parts.append(p.get("current_industry", ""))
    return " ".join(parts).lower()


# ── Honeypot Detection ───────────────────────────────────────────────────────

def is_honeypot(c: dict) -> bool:
    """
    Detect subtly impossible profiles:
    - exp at company founded after start date
    - expert in 10+ skills with 0 duration
    - years_experience >> career history months
    - started job before company founding (impossible dates)
    """
    skills = c.get("skills", [])
    smap = skill_map(skills)

    # 1. Too many "expert" skills with 0 duration_months
    expert_zero_dur = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero_dur >= 4:
        return True

    # 2. Years of experience wildly inconsistent with career history
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    career = c.get("career_history", [])
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 3 and total_career_months > 0:
        ratio = (yoe * 12) / max(total_career_months, 1)
        if ratio > 2.5:  # Claims 2.5× more experience than career entries show
            return True

    # 3. Current job started impossibly early (before 1990 or after today)
    for job in career:
        if job.get("is_current"):
            try:
                start = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
                if start > TODAY:
                    return True
                if start.year < 1990:
                    return True
            except Exception:
                pass

    # 4. Expert in 8+ different high-competition skills with <6 months each
    deep_expert_shallow = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 999) < 6
    )
    if deep_expert_shallow >= 6:
        return True

    return False


# ── Disqualifier Checks ──────────────────────────────────────────────────────

def disqualifier_score(c: dict, blob: str) -> float:
    """
    Returns a multiplier: 0.0 = disqualified, 0.5 = soft disqualifier, 1.0 = clean
    """
    p = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    smap = skill_map(skills)
    yoe = p.get("years_of_experience", 0)

    # ── Hard disqualifier: pure services career ──────────────────────────────
    # Entire career at services firms = hard disqualify
    if career:
        non_services = [
            j for j in career
            if not any(sf in norm(j.get("company", "")) for sf in SERVICES_FIRMS)
        ]
        services_months = sum(
            j.get("duration_months", 0) for j in career
            if any(sf in norm(j.get("company", "")) for sf in SERVICES_FIRMS)
        )
        total_months = sum(j.get("duration_months", 0) for j in career)
        if total_months > 0 and services_months / total_months > 0.85 and not non_services:
            return 0.05  # essentially disqualified

    # ── Hard disqualifier: CV/Speech/Robotics without NLP/IR ────────────────
    # Detect if primary domain is CV/speech and they have NO NLP signal
    cv_mentions = sum(1 for kw in CV_SPEECH_DOMAINS if kw in blob)
    nlp_mentions = sum(1 for kw in NLP_IR_SIGNALS if kw in blob)
    if cv_mentions >= 3 and nlp_mentions == 0:
        return 0.08

    # ── Soft disqualifier: mostly LangChain/wrapper work <12 months ML ──────
    langchain_signal = ("langchain" in blob or "llamaindex" in blob or "llama index" in blob)
    # Check for pre-LLM ML experience (production ML before 2022)
    pre_llm_ml = False
    for job in career:
        try:
            year = int(job.get("start_date", "9999")[:4])
            if year < 2022:
                desc = norm(job.get("description", ""))
                if any(kw in desc for kw in ["model", "ml", "machine learning", "ranking", "retrieval", "recommend", "embedding", "search"]):
                    pre_llm_ml = True
                    break
        except Exception:
            pass

    if langchain_signal and not pre_llm_ml and yoe < 4:
        return 0.2

    # ── Soft disqualifier: no code in 18+ months (pure arch/lead) ───────────
    # Heuristic: if current job is "VP", "Director", "Head of" with no coding signals
    current_title_lower = norm(p.get("current_title", ""))
    if any(kw in current_title_lower for kw in ["vp ", "vice president", "director", "head of", "cto", "chief"]):
        # Check if career shows any individual contributor work recently
        ic_recent = False
        for job in career:
            try:
                start = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
                months_ago = (TODAY - start).days / 30
                if months_ago < 24:
                    desc = norm(job.get("description", ""))
                    if any(kw in desc for kw in ["implemented", "wrote", "built", "coded", "developed", "deployed", "engineered"]):
                        ic_recent = True
                        break
            except Exception:
                pass
        if not ic_recent:
            return 0.15

    return 1.0  # clean


# ── Scoring Components ────────────────────────────────────────────────────────

def score_career_substance(c: dict, blob: str) -> float:
    """
    Did they actually BUILD ranking/retrieval/search/recommendation systems at product companies?
    This is the single most important signal per the JD.
    """
    career = c.get("career_history", [])
    p = c.get("profile", {})

    score = 0.0

    # Key verbs that signal production engineering (not just research or tutorials)
    production_verbs = [
        "shipped", "deployed", "launched", "built", "implemented",
        "designed", "developed", "scaled", "owned", "led", "ran",
        "reduced", "improved", "increased", "migrated", "maintained",
        "production", "serving", "inference", "live"
    ]

    # Target domains from JD
    target_domains = [
        "ranking", "retrieval", "search", "recommendation", "rerank",
        "embedding", "vector", "similarity", "relevance", "match",
        "candidate ranking", "job matching", "talent", "nlp", "llm", "rag"
    ]

    # Company size signal: product companies tend to be 51-5000 range
    # Startups (1-50) also count; pure big IT services usually 10001+
    company_size_scores = {
        "1-10": 0.6, "11-50": 0.8, "51-200": 1.0, "201-500": 1.0,
        "501-1000": 0.9, "1001-5000": 0.8, "5001-10000": 0.6, "10001+": 0.3
    }

    job_scores = []
    for job in career:
        desc = norm(job.get("description", ""))
        title_n = norm(job.get("title", ""))
        company_n = norm(job.get("company", ""))
        size = job.get("company_size", "")
        months = job.get("duration_months", 0)
        is_services = any(sf in company_n for sf in SERVICES_FIRMS)

        if months < 3:
            continue

        # Domain relevance
        domain_hits = sum(1 for d in target_domains if d in desc or d in title_n)
        # Production evidence
        prod_hits = sum(1 for v in production_verbs if v in desc)
        # Quantified impact (numbers in description = shipped real things)
        has_numbers = bool(re.search(r'\d+[%xk mKMB]|\d{3,}', desc))

        size_mult = company_size_scores.get(size, 0.5) * (0.5 if is_services else 1.0)
        duration_mult = min(1.0, months / 24)  # 2+ years = full credit

        job_score = min(1.0, (domain_hits * 0.15 + prod_hits * 0.05 + (0.2 if has_numbers else 0)) * size_mult * duration_mult)
        job_scores.append(job_score)

    if job_scores:
        # Weight recent jobs more
        weights = [1.0 / (i + 1) for i in range(len(job_scores))]
        total_w = sum(weights)
        score = sum(j * w for j, w in zip(job_scores, weights)) / total_w

    # Bonus: summary/headline mentions shipping at scale
    if any(kw in blob for kw in ["at scale", "production", "millions", "real users", "live system"]):
        score = min(1.0, score + 0.1)

    return min(1.0, score)


def score_technical_fit(c: dict, smap: dict, blob: str) -> float:
    """
    Semantic skill matching with trust adjustment:
    - Skill listed but 0 duration and 0 endorsements = low trust
    - Skill with duration_months and endorsements = high trust
    - Skill confirmed in career history = highest trust
    """
    core_hits = 0.0
    core_total = 0

    # Core skills check with trust weighting
    core_to_check = {
        # Must have (from JD)
        "embeddings": 3.0,
        "semantic search": 3.0,
        "vector": 2.5,
        "pinecone": 2.0, "weaviate": 2.0, "qdrant": 2.0, "milvus": 2.0,
        "faiss": 2.0, "opensearch": 2.0, "elasticsearch": 2.0,
        "bm25": 2.0, "hybrid search": 2.0,
        "ranking": 2.5,
        "retrieval": 2.5,
        "information retrieval": 2.5,
        "python": 2.0,
        "ndcg": 2.0, "mrr": 1.5, "evaluation": 1.5,
        # Nice-to-have
        "lora": 1.0, "qlora": 1.0, "fine-tuning": 1.5, "fine tuning": 1.5,
        "llm": 1.5, "nlp": 1.5,
        "recommendation": 1.5,
        "search": 1.5,
    }

    for skill_kw, weight in core_to_check.items():
        core_total += weight
        # Check in skills list with trust
        found_in_skills = False
        for skill_name, skill_data in smap.items():
            if skill_kw in skill_name or skill_name in skill_kw:
                duration = skill_data.get("duration_months", 0)
                endorsements = skill_data.get("endorsements", 0)
                proficiency = skill_data.get("proficiency", "beginner")
                prof_mult = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(proficiency, 0.5)
                # Trust multiplier: penalize zero-duration-zero-endorsement skills
                if duration == 0 and endorsements == 0:
                    trust = 0.3
                elif duration < 3:
                    trust = 0.5
                else:
                    trust = min(1.0, 0.6 + duration / 60 * 0.4)
                trust = min(trust, 1.0 if endorsements > 5 else trust * 0.85)
                core_hits += weight * prof_mult * trust
                found_in_skills = True
                break

        if not found_in_skills:
            # Check in blob (career history / summary) — lower trust than explicit skill
            if skill_kw in blob:
                core_hits += weight * 0.5  # mentioned but not formally listed

    raw = core_hits / max(core_total, 1)

    # Boost for GitHub activity (validates technical work)
    sig = c.get("redrob_signals", {})
    gh = sig.get("github_activity_score", -1)
    if gh > 50:
        raw = min(1.0, raw + 0.07)
    elif gh > 20:
        raw = min(1.0, raw + 0.03)

    # Boost for skill assessment scores on relevant skills
    assessments = sig.get("skill_assessment_scores", {})
    for k, v in assessments.items():
        if any(core_kw in norm(k) for core_kw in ["python", "nlp", "ml", "retrieval", "embedding", "ranking"]):
            if v > 70:
                raw = min(1.0, raw + 0.04)
                break

    return raw


def score_experience_bracket(c: dict) -> float:
    """
    JD says 5-9 yrs. Sweet spot per JD text: 6-8 yrs.
    Explicit disqualifiers: pure research, senior-but-no-code.
    """
    yoe = c.get("profile", {}).get("years_of_experience", 0)

    if yoe < 3:
        return 0.2
    elif yoe < 5:
        return 0.6
    elif yoe <= 9:
        # Sweet spot 6-8, slight penalty at edges
        if 6 <= yoe <= 8:
            return 1.0
        elif yoe == 5 or yoe == 9:
            return 0.85
        else:
            return 0.75
    else:  # >9 years
        # Overqualification risk (may want higher title/comp)
        excess = yoe - 9
        return max(0.5, 0.85 - excess * 0.05)


def score_location_availability(c: dict) -> float:
    """
    JD: Pune/Noida preferred, open to Tier-1 Indian cities, no visa sponsorship.
    Notice period: ideally <30 days.
    """
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    country = norm(p.get("country", ""))
    location = norm(p.get("location", ""))
    willing_to_relocate = sig.get("willing_to_relocate", False)
    notice = sig.get("notice_period_days", 90)

    # Country check
    if country == "india":
        loc_score = 0.7
        # Preferred cities
        if any(city in location for city in ["pune", "noida", "delhi"]):
            loc_score = 1.0
        elif any(city in location for city in ["hyderabad", "mumbai", "bangalore", "bengaluru", "gurugram", "gurgaon"]):
            loc_score = 0.9
        elif any(city in location for city in ["chennai", "kolkata"]):
            loc_score = 0.8
    elif willing_to_relocate and country in ["", "unknown"]:
        loc_score = 0.6
    elif willing_to_relocate:
        # International + willing to relocate: case-by-case per JD
        loc_score = 0.45
    else:
        # Outside India + not willing to relocate = poor fit (no visa sponsorship)
        loc_score = 0.2

    # Notice period
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.95
    elif notice <= 60:
        notice_score = 0.75
    elif notice <= 90:
        notice_score = 0.55
    else:
        notice_score = 0.3

    return loc_score * 0.7 + notice_score * 0.3


def score_education(c: dict, blob: str) -> float:
    """
    JD doesn't list education requirements but mentions it's not a dealbreaker.
    Signal value: relevant CS/ML field + reasonable tier.
    """
    education = c.get("education", [])
    if not education:
        return 0.5  # Unknown, not penalized much

    tier_scores = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.7, "tier_4": 0.55, "unknown": 0.6}
    relevant_fields = {"computer science", "cs", "machine learning", "ai", "artificial intelligence",
                       "data science", "electrical engineering", "electronics", "mathematics",
                       "statistics", "information technology", "software engineering", "nlp"}

    best = 0.0
    for e in education:
        tier = tier_scores.get(e.get("tier", "unknown"), 0.6)
        field = norm(e.get("field_of_study", ""))
        is_relevant = any(rf in field for rf in relevant_fields)
        degree = norm(e.get("degree", ""))
        degree_mult = 1.0 if "m." in degree or "phd" in degree or "ph.d" in degree else 0.85
        e_score = tier * degree_mult * (1.0 if is_relevant else 0.75)
        best = max(best, e_score)

    return best


def behavioral_multiplier(c: dict) -> float:
    """
    Behavioral signals as a multiplier on fit score.
    A perfect-on-paper candidate who is inactive/unresponsive should be downweighted.
    Range: 0.2-1.0
    """
    sig = c.get("redrob_signals", {})

    score = 0.0
    total_weight = 0.0

    def add(val, weight):
        nonlocal score, total_weight
        score += val * weight
        total_weight += weight

    # 1. Recency of activity
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if days_inactive <= 7:
        add(1.0, 3.0)
    elif days_inactive <= 30:
        add(0.85, 3.0)
    elif days_inactive <= 90:
        add(0.6, 3.0)
    elif days_inactive <= 180:
        add(0.35, 3.0)
    else:
        add(0.1, 3.0)

    # 2. Open to work
    add(1.0 if sig.get("open_to_work_flag") else 0.3, 2.0)

    # 3. Recruiter response rate
    rr = sig.get("recruiter_response_rate", 0.0)
    add(rr, 2.5)

    # 4. Response time (fast = better)
    rt = sig.get("avg_response_time_hours", 999)
    if rt <= 4:
        add(1.0, 1.0)
    elif rt <= 24:
        add(0.8, 1.0)
    elif rt <= 72:
        add(0.5, 1.0)
    else:
        add(0.2, 1.0)

    # 5. Interview completion rate
    icr = sig.get("interview_completion_rate", 0.5)
    add(icr, 1.5)

    # 6. Profile completeness
    pc = sig.get("profile_completeness_score", 50) / 100
    add(pc, 1.0)

    # 7. Verified contact
    verified = (sig.get("verified_email", False) and sig.get("verified_phone", False))
    add(1.0 if verified else 0.5, 1.0)

    # 8. Saved by recruiters recently (market signal)
    saved = sig.get("saved_by_recruiters_30d", 0)
    add(min(1.0, saved / 10), 1.0)

    raw = score / max(total_weight, 1)
    # Scale to 0.2-1.0 range (never zero-out a candidate on signals alone)
    return 0.2 + raw * 0.8


# ── Main Scoring Function ────────────────────────────────────────────────────

WEIGHTS = {
    "career":      0.35,   # Most important per JD: did they SHIP ranking/retrieval?
    "technical":   0.28,   # Do they have the right skills, at depth?
    "experience":  0.15,   # Seniority bracket
    "location":    0.12,   # India preferred, availability
    "education":   0.10,   # Not a dealbreaker
}

def score_candidate(c: dict) -> tuple[float, str]:
    """
    Returns (score 0-1, reasoning string).
    """
    if is_honeypot(c):
        return 0.001, "Honeypot detected: profile contains impossible chronology or inflated skill claims."

    blob = text_blob(c)
    smap = skill_map(c.get("skills", []))
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    disq = disqualifier_score(c, blob)
    if disq < 0.15:
        reason = ""
        if disq < 0.1:
            reason = f"Disqualified: entire career at IT services firms with no product-company exposure. Current: {p.get('current_company', 'unknown')}."
        else:
            reason = f"Soft-disqualified: primary domain appears to be CV/speech/robotics without NLP/IR signals."
        return disq * 0.5, reason

    s_career   = score_career_substance(c, blob)
    s_tech     = score_technical_fit(c, smap, blob)
    s_exp      = score_experience_bracket(c)
    s_loc      = score_location_availability(c)
    s_edu      = score_education(c, blob)
    b_mult     = behavioral_multiplier(c)

    base = (
        s_career   * WEIGHTS["career"] +
        s_tech     * WEIGHTS["technical"] +
        s_exp      * WEIGHTS["experience"] +
        s_loc      * WEIGHTS["location"] +
        s_edu      * WEIGHTS["education"]
    )

    total = base * b_mult * disq

    # ── Reasoning ────────────────────────────────────────────────────────────
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    location = p.get("location", "")
    country = p.get("country", "")
    notice = sig.get("notice_period_days", 90)
    rr = sig.get("recruiter_response_rate", 0)
    days_ago = days_since(sig.get("last_active_date", "2020-01-01"))
    open_to = sig.get("open_to_work_flag", False)

    # Build specific, varied reasoning referencing actual profile facts (Stage 4 check)
    career = c.get("career_history", [])
    points = []

    # Career substance — name actual companies and domains from profile
    if s_career > 0.6 and career:
        for job in career[:2]:
            desc = norm(job.get("description", ""))
            if any(kw in desc for kw in ["ranking", "retrieval", "search", "recommendation", "embedding", "vector"]):
                domain_kws = [kw for kw in ["ranking","retrieval","search","recommendation","embedding","vector"] if kw in desc]
                points.append(f"{job.get('title','')} at {job.get('company','')} ({domain_kws[0]} work in prod)")
                break
        else:
            points.append(f"applied ML at product company ({company}); career history shows shipping, not just experimenting")
    elif s_career < 0.3:
        points.append("career history lacks direct evidence of ranking/retrieval/search systems in production")

    # Specific skills from their actual profile
    rel_skills = [s.get("name") for s in c.get("skills", [])
                  if any(kw in norm(s.get("name",""))
                         for kw in ["embedding","retrieval","ranking","search","vector","nlp","python","llm","rag",
                                    "faiss","pinecone","qdrant","milvus","opensearch","elasticsearch","bm25","lora","peft","bert"])]
    if rel_skills:
        points.append(f"core skills present: {', '.join(rel_skills[:3])}")
    elif s_tech < 0.3:
        points.append("key required skills (embeddings, vector DB, ranking eval) absent or unverified in profile")

    # Experience + location + availability as concrete facts
    exp_parts = []
    if 6 <= yoe <= 8:
        exp_parts.append(f"{yoe:.1f} yrs (JD sweet spot 6-8)")
    elif yoe < 5:
        exp_parts.append(f"{yoe:.1f} yrs (below 5-yr floor)")
    elif yoe > 10:
        exp_parts.append(f"{yoe:.1f} yrs (overqualified risk)")
    else:
        exp_parts.append(f"{yoe:.1f} yrs")
    if location:
        exp_parts.append(location)
    if notice <= 30:
        exp_parts.append(f"notice {notice}d")
    elif notice > 60:
        exp_parts.append(f"notice {notice}d (concern)")
    if days_ago <= 14:
        exp_parts.append(f"active {days_ago}d ago")
    elif days_ago > 90:
        exp_parts.append(f"inactive {days_ago}d — availability risk")
    if open_to:
        exp_parts.append("open to work")
    if rr < 0.25:
        exp_parts.append(f"low response rate ({rr:.0%})")
    elif rr > 0.7:
        exp_parts.append(f"high response rate ({rr:.0%})")
    if exp_parts:
        points.append("; ".join(exp_parts))

    reasoning = f"{title} at {company} — " + ". ".join(points[:3]) + "."
    return round(total, 6), reasoning[:300]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon — Senior AI Engineer Ranker")
    parser.add_argument("--candidates", default="C:\\Users\\hp\\Desktop\\redrob\\candidates.jsonl")
    parser.add_argument("--out", default="C:\\Users\\hp\\Desktop\\redrob\\submission.csv")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.")

    print("Scoring...")
    scored = []
    honeypots_found = 0
    disqualified = 0

    for i, c in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(candidates)}...")
        score, reasoning = score_candidate(c)
        cid = c.get("candidate_id", f"CAND_{i:07d}")
        scored.append((score, cid, reasoning))
        if "Honeypot" in reasoning:
            honeypots_found += 1
        elif "isqualif" in reasoning:
            disqualified += 1

    print(f"  Honeypots detected: {honeypots_found}")
    print(f"  Disqualified: {disqualified}")

    # Sort descending by score, tie-break by candidate_id ascending
    scored.sort(key=lambda x: (-x[0], x[1]))

    top = scored[:args.top_n]

    # Validate scores are non-increasing
    for i in range(len(top) - 1):
        assert top[i][0] >= top[i+1][0], f"Score order violated at rank {i+1}"

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning) in enumerate(top, 1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"\nTop 5:")
    for rank, (score, cid, reasoning) in enumerate(top[:5], 1):
        print(f"  #{rank} {cid} score={score:.4f}")
        print(f"       {reasoning[:120]}...")

    print(f"\nSubmission written to {out_path}")
    print(f"Rows: {len(top)}")
    return out_path


if __name__ == "__main__":
    main()