"""Reference deep-research workflow script templates.

The exported builders return workflow source code that is authored into the
workflow runtime.  The full script is the issue #792 live-demo workload; tests use
``build_deep_research_fixture_script`` to exercise the same strange-loop shape
with a small deterministic fan-out and simulated child returns.
"""

from __future__ import annotations

from textwrap import dedent

SCOUT_ANGLES: tuple[str, ...] = (
    "direct-factual",
    "recent-developments",
    "primary-source",
    "contrarian-counter-evidence",
    "academic",
    "statistical",
    "historical-context",
    "stakeholder",
    "technical-deep-dive",
    "regulatory-legal",
)


def build_deep_research_script(
    *,
    scout_agent_id: str = "scout",
    reader_agent_id: str = "reader",
    synthesis_agent_id: str = "synthesis",
    critic_agent_id: str = "critic",
    angles: tuple[str, ...] = SCOUT_ANGLES,
    top_k: int = 16,
    supplementary_limit: int = 4,
    failing_agent_id: str = "scout_nonexistent_deep_research_parity",
) -> str:
    """Return the full deep-research parity workflow source.

    The defaults match the resource names used in the live demo.  Tests pass real
    generated agent ids while preserving the script shape.
    """

    return _render_script(
        scout_agent_id=scout_agent_id,
        reader_agent_id=reader_agent_id,
        synthesis_agent_id=synthesis_agent_id,
        critic_agent_id=critic_agent_id,
        angles=angles,
        top_k=top_k,
        supplementary_limit=supplementary_limit,
        failing_agent_id=failing_agent_id,
    )


def build_deep_research_fixture_script(
    *,
    scout_agent_id: str,
    reader_agent_id: str,
    synthesis_agent_id: str,
    critic_agent_id: str,
    failing_agent_id: str = "scout_nonexistent_fixture",
) -> str:
    """Return the CI fixture variant: 2 scouts, 2 readers, no deferral."""

    return _render_script(
        scout_agent_id=scout_agent_id,
        reader_agent_id=reader_agent_id,
        synthesis_agent_id=synthesis_agent_id,
        critic_agent_id=critic_agent_id,
        angles=("direct-factual", "contrarian-counter-evidence"),
        top_k=2,
        supplementary_limit=0,
        failing_agent_id=failing_agent_id,
    )


def _render_script(
    *,
    scout_agent_id: str,
    reader_agent_id: str,
    synthesis_agent_id: str,
    critic_agent_id: str,
    angles: tuple[str, ...],
    top_k: int,
    supplementary_limit: int,
    failing_agent_id: str,
) -> str:
    return (
        dedent(
            f"""
        import hashlib
        import json
        import urllib.parse

        SCOUT_AGENT_ID = {scout_agent_id!r}
        READER_AGENT_ID = {reader_agent_id!r}
        SYNTHESIS_AGENT_ID = {synthesis_agent_id!r}
        CRITIC_AGENT_ID = {critic_agent_id!r}
        FAILING_AGENT_ID = {failing_agent_id!r}
        ANGLES = {list(angles)!r}
        TOP_K = {top_k!r}
        SUPPLEMENTARY_LIMIT = {supplementary_limit!r}

        SCOUT_SCHEMA = {{
            "type": "object",
            "additionalProperties": False,
            "required": ["angle", "findings", "dead_ends"],
            "properties": {{
                "angle": {{"type": "string"}},
                "findings": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "source_url", "source_title", "confidence"],
                        "properties": {{
                            "claim": {{"type": "string"}},
                            "source_url": {{"type": "string"}},
                            "source_title": {{"type": "string"}},
                            "confidence": {{"type": "number", "minimum": 0, "maximum": 1}},
                        }},
                    }},
                }},
                "dead_ends": {{"type": "array", "items": {{"type": "string"}}}},
            }},
        }}

        READER_SCHEMA = {{
            "type": "object",
            "additionalProperties": False,
            "required": ["source_url", "credibility", "key_facts"],
            "properties": {{
                "source_url": {{"type": "string"}},
                "credibility": {{"type": "number", "minimum": 0, "maximum": 1}},
                "key_facts": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["fact", "confidence"],
                        "properties": {{
                            "fact": {{"type": "string"}},
                            "quote": {{"type": "string"}},
                            "confidence": {{"type": "number", "minimum": 0, "maximum": 1}},
                        }},
                    }},
                }},
            }},
        }}

        SYNTHESIS_SCHEMA = {{
            "type": "object",
            "additionalProperties": False,
            "required": ["report_markdown", "citations", "open_questions"],
            "properties": {{
                "report_markdown": {{"type": "string"}},
                "citations": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["n", "url", "title"],
                        "properties": {{
                            "n": {{"type": "integer"}},
                            "url": {{"type": "string"}},
                            "title": {{"type": "string"}},
                        }},
                    }},
                }},
                "open_questions": {{"type": "array", "items": {{"type": "string"}}}},
            }},
        }}

        CRITIC_SCHEMA = {{
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "gaps"],
            "properties": {{
                "verdict": {{"type": "string", "enum": ["complete", "gaps"]}},
                "gaps": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["description", "suggested_angle"],
                        "properties": {{
                            "description": {{"type": "string"}},
                            "suggested_angle": {{"type": "string"}},
                        }},
                    }},
                }},
            }},
        }}

        OUTPUT_SCHEMA = {{
            "type": "object",
            "required": ["report", "citations", "critic_verdict", "stats"],
            "properties": {{
                "report": {{"type": "string"}},
                "citations": {{"type": "array"}},
                "critic_verdict": {{"type": "object"}},
                "stats": {{"type": "object"}},
            }},
        }}

        def normalize_url(url):
            parsed = urllib.parse.urlparse(url or "")
            scheme = (parsed.scheme or "https").lower()
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            path = parsed.path.rstrip("/") or "/"
            query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query)))
            return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))

        def claim_key(text):
            normalized = " ".join((text or "").lower().split())
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        def merge_scout_results(results):
            sources = {{}}
            claims = {{}}
            for result in results:
                if not result:
                    continue
                for finding in result.get("findings", []):
                    url = normalize_url(finding.get("source_url", ""))
                    if not url:
                        continue
                    source = sources.setdefault(url, {{
                        "url": url,
                        "title": finding.get("source_title") or url,
                        "confidence_total": 0.0,
                        "corroboration": 0,
                        "claims": [],
                    }})
                    source["confidence_total"] += float(finding.get("confidence") or 0)
                    source["corroboration"] += 1
                    key = claim_key(finding.get("claim", ""))
                    if key not in claims:
                        claims[key] = finding.get("claim", "")
                        source["claims"].append(claims[key])
            ranked = sorted(
                sources.values(),
                key=lambda s: (s["confidence_total"] * max(1, s["corroboration"]), s["url"]),
                reverse=True,
            )
            return ranked, claims

        async def failed_scout():
            try:
                await agent({{"expected_failure": True}}, agent_id=FAILING_AGENT_ID, output_schema=SCOUT_SCHEMA)
            except AgentError as e:
                log(f"scout failed (expected): {{e}}")
                return None
            return None

        def scout_thunk(angle):
            return lambda: agent({{"question": QUESTION, "angle": angle}}, agent_id=SCOUT_AGENT_ID, output_schema=SCOUT_SCHEMA)

        def read_stage(source):
            return agent({{"question": QUESTION, "source": source}}, agent_id=READER_AGENT_ID, output_schema=READER_SCHEMA)

        def normalize_stage(reading, source, index):
            if not reading:
                return None
            reading["source_url"] = normalize_url(reading.get("source_url") or source.get("url"))
            reading["rank"] = index + 1
            return reading

        async def synthesize(question, sources, readings):
            return await agent(
                {{"question": question, "sources": sources, "readings": readings}},
                agent_id=SYNTHESIS_AGENT_ID,
                output_schema=SYNTHESIS_SCHEMA,
            )

        async def critique(question, draft):
            return await agent({{"question": question, "draft": draft}}, agent_id=CRITIC_AGENT_ID, output_schema=CRITIC_SCHEMA)

        async def supplementary_scout(gap):
            return await agent(
                {{"question": QUESTION, "angle": gap.get("suggested_angle", "supplementary"), "gap": gap}},
                agent_id=SCOUT_AGENT_ID,
                output_schema=SCOUT_SCHEMA,
            )

        async def main(input):
            global QUESTION
            QUESTION = input["question"]
            stats = {{"scouts": 0, "sources": 0, "readers": 0, "supplementary_scouts": 0}}

            phase("Phase 1: sweep")
            scout_results = await parallel([scout_thunk(angle) for angle in ANGLES] + [failed_scout])
            stats["scouts"] = len([r for r in scout_results if r])
            sources, claims = merge_scout_results(scout_results)
            stats["sources"] = len(sources)
            log("deduped", len(claims), "claims from", len(sources), "sources")

            phase("Phase 2: deep read")
            survivors = sources[:TOP_K]
            readings = [r for r in await pipeline(survivors, read_stage, normalize_stage) if r]
            stats["readers"] = len(readings)
            log("read", len(readings), "sources")

            phase("Phase 3: synthesis")
            draft = await synthesize(QUESTION, survivors, readings)
            verdict = await critique(QUESTION, draft)

            if input.get("pause_for_review"):
                review = await gate({{"draft_report": draft, "gaps": verdict.get("gaps", [])}})
                if isinstance(review, dict) and review.get("veto_supplementary"):
                    verdict = {{"verdict": verdict.get("verdict", "gaps"), "gaps": [], "review": review}}

            gaps = verdict.get("gaps", [])[:SUPPLEMENTARY_LIMIT]
            if gaps:
                extra = await parallel([lambda gap=gap: supplementary_scout(gap) for gap in gaps])
                stats["supplementary_scouts"] = len([r for r in extra if r])
                extra_sources, _extra_claims = merge_scout_results(extra)
                sources = sources + [s for s in extra_sources if s["url"] not in {{src["url"] for src in sources}}]
                draft = await synthesize(QUESTION, sources[:TOP_K], readings)
                verdict = await critique(QUESTION, draft)

            return {{
                "report": draft.get("report_markdown", ""),
                "citations": draft.get("citations", []),
                "critic_verdict": verdict,
                "stats": stats,
            }}
        """
        ).strip()
        + "\n"
    )
