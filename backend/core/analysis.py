"""
backend/core/analysis.py
Core analysis classes ported from src/analysis.py.
Stripped of all Streamlit, Plotly, Folium dependencies.
"""

import re
import time
import logging
from collections import Counter
from datetime import datetime

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

MIN_YEAR = 2014


# ── QueryBuilder ──────────────────────────────────────────────────────────────

class QueryBuilder:
    # Convention: a term's `operator` sits BEFORE that term in the serialized
    # query (it connects term N-1 → term N). This matches the frontend UI,
    # where the operator shown between two cards is stored on the RIGHT card.
    # The first non-empty term has no preceding operator.

    @staticmethod
    def build_google_query(topics, min_year=None, include_patents=True):
        """Construye la consulta para SerpAPI/Google respetando operadores booleanos."""
        if not topics:
            return ""

        query_parts = []
        for topic in topics:
            if not topic["value"].strip():
                continue

            term = topic["value"].strip()
            if topic.get("exact_match") and not (term.startswith('"') and term.endswith('"')):
                term = f'"{term}"'

            if not query_parts:
                query_parts.append(term)
            else:
                op = topic.get("operator", "AND")
                query_parts.append(f"{op} {term}")

        base_query = " ".join(query_parts)

        filters = []
        if min_year:
            filters.append(f"after:{min_year}")
        if not include_patents:
            filters.append("-patent")

        if filters:
            base_query += " " + " ".join(filters)

        return base_query

    @staticmethod
    def build_scopus_query(topics, min_year=None):
        if not topics:
            return ""

        query_parts = []
        for topic in topics:
            if not topic["value"].strip():
                continue

            term = topic["value"].strip()
            if topic.get("exact_match") and not (term.startswith('"') and term.endswith('"')):
                term = f'"{term}"'

            scopus_term = f"TITLE-ABS-KEY({term})"

            if not query_parts:
                query_parts.append(scopus_term)
            else:
                op = topic.get("operator", "AND")
                if op == "NOT":
                    op = "AND NOT"
                query_parts.append(f"{op} {scopus_term}")

        base_query = " ".join(query_parts)
        if min_year:
            base_query += f" AND PUBYEAR > {min_year}"

        return base_query


# ── ResultAnalyzer ────────────────────────────────────────────────────────────

class ResultAnalyzer:
    def __init__(self):
        self.common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "was", "is", "are", "were", "been",
            "based", "using", "since", "more", "has", "such", "may",
            "very", "both", "each", "between", "under", "same", "through",
            "until",
        }

    def extract_year(self, text):
        current_year = datetime.now().year
        MAX_VALID_YEAR = min(current_year, 2025)
        MIN_VALID_YEAR = 1970

        def is_valid_year(year):
            try:
                return MIN_VALID_YEAR <= int(year) <= MAX_VALID_YEAR
            except ValueError:
                return False

        cleaned_text = text.lower()
        cleaned_text = re.sub(r"\d+\s*(?:kb|mb|gb|kib|mib|gib|bytes?)", "", cleaned_text)
        cleaned_text = re.sub(r"(?:page|p\.)\s*\d+|\d+\s*(?:pages?|p\.)", "", cleaned_text)

        date_patterns = [
            r"published.*?in\s*(20\d{2})",
            r"publication\s*date:?\s*(20\d{2})",
            r"©\s*(20\d{2})",
            r"\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(20\d{2})",
        ]
        for pattern in date_patterns:
            for match in re.findall(pattern, cleaned_text):
                if is_valid_year(match):
                    return int(match)

        year_pattern = r"\b(19[7-9]\d|20[0-2]\d)\b"
        years = re.findall(year_pattern, cleaned_text)
        if years:
            valid_years = [int(y) for y in years if is_valid_year(y)]
            if valid_years:
                return max(valid_years)

        return min(current_year, MAX_VALID_YEAR)

    def classify_result_type(self, url, title):
        url_lower = url.lower()
        title_lower = title.lower()
        if any(t in url_lower for t in [".pdf", "/pdf/"]):
            return "PDF"
        elif any(t in url_lower for t in ["patent", "uspto.gov", "espacenet"]):
            return "Patent"
        elif any(t in url_lower for t in ["scholar.google", "sciencedirect", "springer", "ieee"]):
            return "Academic Article"
        elif any(t in url_lower for t in ["news.google", "reuters", "bloomberg"]):
            return "News"
        elif "cite" in url_lower or "citation" in title_lower:
            return "Citation"
        else:
            return "Web Page"

    def analyze_results(self, results, search_topics):
        search_words: set = set()
        for topic in search_topics:
            words = re.sub(r"[^\w\s]", " ", (topic["value"] if isinstance(topic, dict) else str(topic)).lower()).split()
            search_words.update(words)

        processed_results = []
        all_keywords = []

        for item in results:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            full_text = f"{title}. {snippet}"

            keywords = [
                w for w in re.sub(r"[^\w\s]", " ", full_text.lower()).split()
                if (
                    w not in self.common_words
                    and w not in search_words
                    and len(w) > 3
                    and w.isalpha()
                    and not re.match(r"20\d{2}", w)
                )
            ]
            all_keywords.extend(keywords)

            processed_results.append({
                "title": title,
                "link": item.get("link", ""),
                "snippet": snippet,
                "year": self.extract_year(full_text),
                "country": None,
                "type": self.classify_result_type(item.get("link", ""), title),
            })

        df = pd.DataFrame(processed_results)
        keyword_counts = Counter(all_keywords)
        filtered_keywords = [
            (w, c) for w, c in keyword_counts.most_common(20) if w not in search_words
        ]

        stats = {
            "total_results": len(df),
            "by_type": df["type"].value_counts().to_dict(),
            "by_year": df["year"].value_counts().sort_index().to_dict(),
            "by_country": {},
            "common_keywords": filtered_keywords,
        }
        return processed_results, stats


# ── NewsAnalyzer ──────────────────────────────────────────────────────────────

class NewsAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.SERP_API_BASE_URL = "https://serpapi.com/search"

    # ── Public API ────────────────────────────────────────────────────────────

    def perform_news_search(self, serp_api_key: str, query: str):
        """
        Hybrid strategy: SIMPLE queries → exploratory; COMPLEX → direct date ranges.
        Returns (success: bool, results: list | error_str)
        """
        try:
            current_year = datetime.now().year
            start_year = current_year - 12
            clean_query = re.sub(r"\s*(?:after|before):\d{4}(?:-\d{2}-\d{2})?\s*", "", query).strip()
            complexity = self._analyze_query_complexity(clean_query)

            logger.info("Search strategy: %s — %s", complexity["level"], complexity["reason"])

            if complexity["level"] == "SIMPLE":
                return self._search_simple_query(serp_api_key, clean_query, start_year, current_year)
            else:
                return self._search_complex_query(serp_api_key, clean_query, start_year, current_year)

        except Exception as e:
            logger.error("Search error: %s", e)
            return False, str(e)

    def analyze_hype_cycle(self, news_results: list):
        """
        Computes phase, confidence, yearly stats and inflection points from news results.
        Returns dict or None if no usable data at all.
        """
        try:
            current_year = datetime.now().year
            df_data = []
            for result in news_results:
                try:
                    raw_year = result.get("year")
                    year = int(raw_year) if raw_year is not None else current_year
                    # Allow years in a broad range; clamp out-of-range to current year
                    if not (2000 <= year <= current_year):
                        year = current_year
                    raw_sent = result.get("sentiment", 0.0)
                    sentiment = float(raw_sent) if raw_sent is not None else 0.0
                    if not (-1 <= sentiment <= 1):
                        sentiment = 0.0
                    df_data.append({
                        "year": year,
                        "sentiment": sentiment,
                        "month": self._extract_month(result.get("date", "")),
                    })
                except (ValueError, TypeError, KeyError) as ex:
                    logger.debug("Skipping result in df_data build: %s", ex)
                    continue

            logger.info("analyze_hype_cycle: %d/%d usable rows", len(df_data), len(news_results))

            if len(df_data) == 0:
                logger.warning("analyze_hype_cycle: 0 usable rows — returning None")
                return None

            df = pd.DataFrame(df_data)
            grouped = df.groupby("year")["sentiment"].agg(["mean", "count", "std", "min", "max"])
            grouped.columns = ["sentiment_mean", "mention_count", "sentiment_std", "sentiment_min", "sentiment_max"]
            yearly_stats = grouped.reset_index().rename(columns={"index": "year"})
            yearly_stats = yearly_stats.sort_values("year").reset_index(drop=True)
            # fillna(0) prevents NaN from leaking into JSON (pct_change/diff produce NaN on row 0,
            # and std() is NaN for groups with a single member)
            yearly_stats["sentiment_std"] = yearly_stats["sentiment_std"].fillna(0)
            yearly_stats["mention_change"] = yearly_stats["mention_count"].pct_change().fillna(0)
            yearly_stats["sentiment_change"] = yearly_stats["sentiment_mean"].diff().fillna(0)
            yearly_stats["momentum"] = (
                yearly_stats["mention_change"] * yearly_stats["sentiment_mean"]
            ).fillna(0)

            inflection_points = self.analyze_gartner_points(yearly_stats)

            current_year = datetime.now().year
            phase = "Pre-Innovation Trigger"
            confidence = 0.5

            if inflection_points["innovation_trigger"]:
                innovation_year = inflection_points["innovation_trigger"]["year"]
                years_since_innovation = current_year - innovation_year

                if years_since_innovation <= 1:
                    phase = "Innovation Trigger"
                    confidence = 0.85

                elif inflection_points["peak"]:
                    peak_year = inflection_points["peak"]["year"]
                    years_since_peak = current_year - peak_year
                    peak_sentiment = inflection_points["peak"]["sentiment"]

                    if years_since_peak <= 1 and peak_sentiment > 0.2:
                        phase = "Peak of Inflated Expectations"
                        confidence = 0.9

                    elif inflection_points["trough"]:
                        trough_year = inflection_points["trough"]["year"]
                        years_since_trough = current_year - trough_year

                        if years_since_trough <= 1:
                            phase = "Trough of Disillusionment"
                            confidence = 0.85
                        elif 1 < years_since_trough <= 4:
                            recent_trend = yearly_stats["mention_change"].tail(2).mean()
                            if recent_trend > 0:
                                phase = "Slope of Enlightenment"
                                confidence = 0.8
                            else:
                                phase = "Trough of Disillusionment"
                                confidence = 0.75
                        elif years_since_trough > 4:
                            recent_tail = yearly_stats["mention_count"].tail(3)
                            recent_std = recent_tail.std(ddof=0) if len(recent_tail) > 1 else 0.0
                            recent_mean = recent_tail.mean()
                            if recent_mean > 0 and (recent_std / recent_mean) < 0.3:
                                phase = "Plateau of Productivity"
                                confidence = 0.85
                            else:
                                phase = "Slope of Enlightenment"
                                confidence = 0.7

                    else:
                        if years_since_peak > 2:
                            phase = "Trough of Disillusionment"
                            confidence = 0.7
                        else:
                            post_peak = yearly_stats[yearly_stats["year"] > peak_year]
                            if not post_peak.empty:
                                trend = post_peak["mention_change"].mean()
                                if trend < -0.2:
                                    phase = "Trough of Disillusionment"
                                    confidence = 0.75
                                else:
                                    phase = "Peak of Inflated Expectations"
                                    confidence = 0.65

                elif years_since_innovation > 1:
                    recent_growth = yearly_stats["mention_change"].tail(2).mean()
                    recent_sentiment = yearly_stats["sentiment_mean"].tail(2).mean()
                    if recent_growth > 0.5 and recent_sentiment > 0.3:
                        phase = "Peak of Inflated Expectations"
                        confidence = 0.7
                    else:
                        phase = "Innovation Trigger"
                        confidence = 0.6
            else:
                latest_data = yearly_stats.iloc[-1]
                growth_trend = yearly_stats["mention_change"].tail(3).mean()
                latest_sentiment = latest_data["sentiment_mean"]

                if growth_trend > 0.3 and latest_sentiment > 0.1:
                    phase = "Innovation Trigger"
                    confidence = 0.6
                elif growth_trend < -0.3:
                    phase = "Trough of Disillusionment"
                    confidence = 0.55
                else:
                    phase = "Pre-Innovation Trigger"
                    confidence = 0.5

            # Serialize yearly_stats to plain JSON-safe dicts
            import math
            def _safe(v):
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v

            yearly_stats_list = [
                {k: _safe(v) for k, v in row.items()}
                for row in yearly_stats.to_dict(orient="records")
            ]

            return {
                "phase": phase,
                "confidence": float(confidence),
                "yearly_stats": yearly_stats_list,
                "inflection_points": inflection_points,
                "metrics": {
                    "latest_year": int(yearly_stats.iloc[-1]["year"]),
                    "total_mentions": int(yearly_stats["mention_count"].sum()),
                    "peak_mentions": int(yearly_stats["mention_count"].max()),
                    "avg_sentiment": float(yearly_stats["sentiment_mean"].mean()),
                    "sentiment_volatility": float(yearly_stats["sentiment_std"].mean()),
                },
            }

        except Exception as e:
            logger.error("analyze_hype_cycle error: %s", e, exc_info=True)
            return None

    def analyze_gartner_points(self, yearly_stats):
        """Identifica innovation trigger, peak y trough desde el dataframe de stats."""
        try:
            df = pd.DataFrame(yearly_stats).sort_values("year")
            df["mention_pct_change"] = df["mention_count"].pct_change()
            df["sentiment_change"] = df["sentiment_mean"].diff()

            inflection_points = {"innovation_trigger": None, "peak": None, "trough": None}

            if not df.empty:
                threshold = df["mention_count"].mean() * 0.1
                trigger_candidates = df[df["mention_count"] >= threshold]
                if trigger_candidates.empty:
                    trigger_candidates = df  # fallback: use all rows
                innovation_row = trigger_candidates.iloc[0]
                inflection_points["innovation_trigger"] = {
                    "year": int(innovation_row["year"]),
                    "mentions": int(innovation_row["mention_count"]),
                    "sentiment": float(innovation_row["sentiment_mean"]),
                }

                peak_row = df.loc[df["mention_count"].idxmax()]
                inflection_points["peak"] = {
                    "year": int(peak_row["year"]),
                    "mentions": int(peak_row["mention_count"]),
                    "sentiment": float(peak_row["sentiment_mean"]),
                }

                post_peak = df[df["year"] > peak_row["year"]]
                if not post_peak.empty:
                    trough_row = post_peak.loc[post_peak["mention_count"].idxmin()]
                    inflection_points["trough"] = {
                        "year": int(trough_row["year"]),
                        "mentions": int(trough_row["mention_count"]),
                        "sentiment": float(trough_row["sentiment_mean"]),
                    }

            return inflection_points

        except Exception as e:
            logger.error("analyze_gartner_points error: %s", e)
            return {"innovation_trigger": None, "peak": None, "trough": None}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _analyze_query_complexity(self, query: str) -> dict:
        complexity_score = 0
        reasons = []

        for op in ["AND", "OR", "NOT", "&", "|", "-"]:
            if op in query.upper():
                complexity_score += 2
                reasons.append(f"operator {op}")

        quoted_terms = re.findall(r'"[^"]+"', query)
        if len(quoted_terms) > 1:
            complexity_score += len(quoted_terms)
            reasons.append(f"{len(quoted_terms)} exact terms")

        words_outside = re.sub(r'"[^"]+"', "", query).strip()
        word_count = len([w for w in words_outside.split() if len(w) > 2])
        if word_count > 2:
            complexity_score += word_count - 2
            reasons.append(f"{word_count} free words")

        for f in ["site:", "filetype:", "intitle:", "inurl:", "after:", "before:"]:
            if f in query.lower():
                complexity_score += 1
                reasons.append(f"filter {f}")

        if complexity_score <= 2:
            level = "SIMPLE"
            reason = "low complexity: " + (", ".join(reasons[:2]) or "single term")
        else:
            level = "COMPLEX"
            reason = "high complexity: " + ", ".join(reasons[:3])

        return {"level": level, "reason": reason, "score": complexity_score}

    def _search_simple_query(self, serp_api_key: str, query: str, start_year: int, current_year: int):
        base_params = {
            "api_key": serp_api_key,
            "tbm": "nws",
            "num": 100,
            "safe": "off",
            "gl": "us",
            "hl": "en",
            "filter": "0",
        }
        all_results = []
        total_api_calls = 0

        # Note: after:/before: date filters are NOT reliable for tbm=nws (Google News).
        # Search without date filter first; results are filtered by year in post-processing.
        params = {**base_params, "q": query, "start": 0}

        try:
            response = requests.get(self.SERP_API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            total_api_calls += 1

            # Log response keys for debugging
            top_keys = list(data.keys())
            logger.info("SerpAPI response keys: %s", top_keys)
            if "error" in data:
                logger.error("SerpAPI error: %s", data["error"])
                return False, f"SerpAPI error: {data['error']}"

            if "news_results" not in data or not data["news_results"]:
                logger.warning("No news_results in response. Keys: %s", top_keys)
                return False, "No results found in exploratory query"

            first_batch = data["news_results"]
            batch_count = len(first_batch)
            logger.info("SerpAPI returned %d news results", batch_count)

            for item in first_batch:
                if self._is_valid_result(item):
                    processed = self._process_news_item(item)
                    if processed:
                        all_results.append(processed)

            logger.info("After validation: %d usable results", len(all_results))

            # If there are exactly 100 results, try page 2
            if batch_count == 100:
                try:
                    r2 = requests.get(self.SERP_API_BASE_URL, params={**base_params, "q": query, "start": 100}, timeout=30)
                    r2.raise_for_status()
                    d2 = r2.json()
                    total_api_calls += 1
                    for item in d2.get("news_results", []):
                        if self._is_valid_result(item):
                            p = self._process_news_item(item)
                            if p:
                                all_results.append(p)
                except Exception:
                    pass

        except Exception as e:
            return False, f"SerpAPI request failed: {e}"

        return True, self._remove_duplicates(all_results)

    def _search_complex_query(self, serp_api_key: str, query: str, start_year: int, current_year: int):
        base_params = {
            "api_key": serp_api_key,
            "tbm": "nws",
            "num": 100,
            "safe": "off",
            "gl": "us",
            "hl": "en",
            "filter": "0",
        }
        all_results = []
        total_api_calls = 0

        date_ranges = [
            (start_year, start_year + 2),
            (start_year + 3, start_year + 5),
            (start_year + 6, start_year + 8),
            (start_year + 9, current_year),
        ]

        for (rs, re_) in date_ranges:
            start = 0
            calls = 0
            no_results = 0

            while calls < 10:
                try:
                    dq = f"{query} after:{rs}-01-01 before:{re_}-12-31"
                    r = requests.get(self.SERP_API_BASE_URL, params={**base_params, "q": dq, "start": start}, timeout=30)
                    r.raise_for_status()
                    d = r.json()
                    total_api_calls += 1
                    calls += 1
                    batch = d.get("news_results", [])

                    if batch:
                        no_results = 0
                        for item in batch:
                            if self._is_valid_result(item):
                                p = self._process_news_item(item)
                                if p:
                                    all_results.append(p)
                        start += len(batch)
                        time.sleep(0.3)
                    else:
                        no_results += 1
                        if no_results >= 2:
                            break
                        start += 100

                except Exception as e:
                    logger.warning("Complex range error: %s", e)
                    no_results += 1
                    if no_results >= 2:
                        break
                    time.sleep(1)
                    start += 100

        return True, self._remove_duplicates(all_results)

    def _remove_duplicates(self, results: list) -> list:
        seen_urls: set = set()
        seen_titles: set = set()
        unique = []
        for r in results:
            url = r.get("link", "")
            title = r.get("title", "").lower().strip()
            if url not in seen_urls and title not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title)
                unique.append(r)
        return unique

    def _is_valid_result(self, item: dict) -> bool:
        try:
            if not all(k in item for k in ["title", "link", "snippet"]):
                return False
            if len(item["snippet"]) < 50:
                return False
            blocked = ["pinterest", "facebook.com", "twitter.com", "instagram.com"]
            if any(d in item["link"].lower() for d in blocked):
                return False
            return True
        except Exception:
            return False

    def _process_news_item(self, item: dict):
        try:
            if not item.get("title") or not item.get("link"):
                return None

            processed = {
                "title": str(item.get("title", "")),
                "link": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
                "source": str(item.get("source", "")),
                "date": str(item.get("date", "")),
                "year": self._extract_year_from_date(item.get("date", "")),
                "thumbnail": str(item.get("thumbnail", "")),
                "keywords": [],
                "sentiment": 0.0,
                "country": None,
            }

            text = f"{processed['title']} {processed['snippet']}"
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                processed["sentiment"] = scores["compound"]
            except Exception:
                processed["sentiment"] = 0.0

            try:
                processed["keywords"] = self._extract_keywords(text)
            except Exception:
                pass

            return processed

        except Exception as e:
            logger.warning("_process_news_item error: %s", e)
            return None

    def _extract_year_from_date(self, date_str: str) -> int:
        try:
            if not date_str:
                return datetime.now().year
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%b %d, %Y", "%Y"]:
                try:
                    return datetime.strptime(date_str.split("T")[0], fmt).year
                except ValueError:
                    continue
            m = re.search(r"20\d{2}|19\d{2}", date_str)
            if m:
                y = int(m.group())
                if 1970 <= y <= datetime.now().year:
                    return y
            return datetime.now().year
        except Exception:
            return datetime.now().year

    def _extract_month(self, date_str: str):
        try:
            if not date_str:
                return None
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%b %d, %Y"]:
                try:
                    return datetime.strptime(date_str.split("T")[0], fmt).month
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _extract_keywords(self, text: str) -> list:
        words = re.findall(r"\b\w+\b", text.lower())
        stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        kw = [w for w in words if w not in stop and len(w) > 3]
        return Counter(kw).most_common(5)
