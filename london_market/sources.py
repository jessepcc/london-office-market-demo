from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal
from urllib.parse import urlsplit

from .domain import Document, Evidence, Geography, Observation, Quantity, SourceReportedChange, ForecastDeliveryRisk, SourceQualification, ComparisonDiscrepancy, PreletCommitment, compatibility
from .providers import FetchedPage


@dataclass(frozen=True)
class ReportedChangeSpec:
    pattern: str
    baseline: tuple[date, date]
    endpoint: tuple[date, date]
    unit: str
    sign: int = -1
    basis: Literal["absolute", "relative_percent"] = "absolute"
    requires_comparison: bool = False


@dataclass(frozen=True)
class ForecastRiskSpec:
    forecast_pattern: str
    risk_pattern: str
    unit: str


QualificationSpec = ReportedChangeSpec | ForecastRiskSpec


@dataclass(frozen=True)
class PreletSpec:
    pattern: str
    period: tuple[date, date]


@dataclass(frozen=True)
class Measurement:
    metric: str
    pattern: str
    geography: Geography
    unit: str
    definition: str
    category: str
    period: tuple[date, date] | None = None
    native_geography: str | None = None
    dimension: str | None = None
    segment: str | None = None
    partition: str | None = None
    weekly_basis: Literal["period_total", "snapshot"] | None = None
    qualifications: tuple[QualificationSpec, ...] = ()


@dataclass(frozen=True)
class Passage:
    category: str
    pattern: str
    native_geography: str
    geography: Geography
    qualification: str = ""
    kind: Literal["fact", "forecast", "interpretation"] | None = None
    facets: tuple[str, ...] = ()
    qualifications: tuple[QualificationSpec, ...] = ()
    prelet: PreletSpec | None = None


@dataclass(frozen=True)
class MonitoredSource:
    id: str
    url: str
    header: str
    published_at: date | None
    period: tuple[date, date] | None
    measurements: tuple[Measurement, ...]
    qualitative_patterns: tuple[tuple[str, str] | Passage, ...]
    caveats: tuple[str, ...]
    version: str = "2"
    publisher: str = "Savills"
    source_kind: str = "primary_research"
    topics: tuple[str, ...] = ("rents", "vacancy_availability", "take_up", "supply_pipeline", "submarket_dynamics", "occupier_demand", "news")
    geographies: tuple[Geography, ...] = ("Central London", "City", "West End")
    discovery_query: str = "Savills Central London Office Market Watch latest quarterly research"
    discovery_urls: tuple[str, ...] = ("https://www.savills.com/research_articles/255800/393427-0",)
    accepted_metadata_dates: tuple[date, ...] = ()
    required_phrases: tuple[str, ...] = ()
    weekly_basis: Literal["period_total", "snapshot"] | None = None
    allow_document_title: bool = False


Q1 = (date(2026, 1, 1), date(2026, 3, 31))
Q2 = (date(2026, 4, 1), date(2026, 6, 30))
H1 = (date(2026, 1, 1), date(2026, 6, 30))
MAY = (date(2026, 5, 1), date(2026, 5, 31))
NUMBER = r"(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)"
TAKEUP = "Office space leased during the reference period, as reported by Savills"
VACANCY = "Savills reported office vacancy rate at reference-period end; not availability"
SUPPLY = "Savills reported office supply at reference-period end; not a vacancy rate"
COMPLETIONS = "Office development and major refurbishment completions during the reference period"
PRIME = "Savills reported average prime rent from leasing transactions; not asking rent or a constant-quality index"
GRADE_A = "Savills reported average Grade A rent from leasing transactions; not asking rent or a constant-quality index"


def m(metric: str, phrase: str, geography: Geography, unit: str, definition: str, category: str, period=None, **dimensions) -> Measurement:
    return Measurement(metric, phrase.replace("{n}", NUMBER), geography, unit, definition, category, period, **dimensions)


MONITORED_SOURCES = {
    "savills-central-london-q1-2026": MonitoredSource(
        "savills-central-london-q1-2026", "https://www.savills.com/research_articles/255800/390337-0",
        "Central London Office Market Watch Q1 2026 01 May 2026 Article", date(2026, 5, 1), Q1,
        (
            m("take_up", r"Leasing across Central London reached {n}m sq ft across \d+ transactions at the end of Q1\.", "Central London", "million sq ft", TAKEUP, "take-up"),
            m("reported_supply", r"Supply across Central London remained stable over the quarter at {n}m sq ft", "Central London", "million sq ft", SUPPLY, "availability"),
            m("vacancy_rate", r"This equates to a vacancy rate of {n}%", "Central London", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"The City vacancy rate remained at the same level at {n}%", "City", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"West End market saw a slight contraction to supply, bringing the vacancy rate down 10 bps on the previous quarter to {n}%", "West End", "%", VACANCY, "vacancy"),
            m("development_completions", r"Development completions reached {n}m sq ft, with nine schemes reaching practical completion by the end of Q1\.", "Central London", "million sq ft", COMPLETIONS, "pipeline"),
            m("prime_rent", r"average prime rent for the City rose to £{n} per sq ft\.", "City", "GBP per sq ft", PRIME, "prime_rent"),
            m("prime_rent", r"average prime rent across the West End held steady at £{n} per sq ft", "West End", "GBP per sq ft", PRIME, "prime_rent"),
        ),
        (
            Passage("take-up", r"The largest pre-let \(and transaction\) to complete was global law firm Herbert Smith Freehills Kramer[^.]{1,500}\.", "Central London", "Central London", facets=("prelets", "leasing"), prelet=PreletSpec(r"The largest pre-let \(and transaction\) to complete was global law firm (?P<occupier>Herbert Smith Freehills Kramer)’s acquisition of British Land and GIC’s (?P<building>1 Appold Street, EC2) \(268,000 sq ft\), at £104 per sq ft, for a term of (?P<term>21) years\.", Q1)),
            Passage("occupier_demand", r"Leasing activity continued to be driven by demand for Grade A office space, which accounted for 92% of Q1 take.up\. This demand[^.]{1,350}\.", "Central London", "Central London", facets=("flight_to_quality",)),
            Passage("pipeline", r"There has been a notable shift towards refurbishments, with comprehensive refurbishments[^.]{1,500}\.", "Central London", "Central London", facets=("refurbishment",)),
        ),
        ("Central London aggregate does not establish whole-London coverage.", "Q1 Grade A rent text says 'at the end of the year'; excluded from normalized observations.", "Prime rents are transaction averages and may reflect deal mix. Q2 rents use an H1 period and cannot form a quarterly trend with Q1."), version="3",
    ),
    "savills-central-london-q2-2026": MonitoredSource(
        "savills-central-london-q2-2026", "https://www.savills.com/research_articles/255800/393427-0",
        "Central London Office Market Q2 2026 06 August 2026 Article", date(2026, 8, 6), Q2,
        (
            m("take_up", r"Central London continued to witness robust levels of occupational demand in Q2, with take-up reaching {n} million sq ft across \d+ transactions\.", "Central London", "million sq ft", TAKEUP, "take-up"),
            m("reported_supply", r"Supply across Central London fell to {n} million sq ft during the quarter, down 0\.5 million sq ft on Q1 2026\.", "Central London", "million sq ft", SUPPLY, "availability", qualifications=(ReportedChangeSpec(r"down (?P<change>\d+(?:\.\d+)?) million sq ft on Q1 2026", Q1, Q2, "million sq ft", requires_comparison=True),)),
            m("vacancy_rate", r"As a result, the vacancy rate declined by 20 bps to {n}%", "Central London", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"The City vacancy rate remained unchanged at {n}%", "City", "%", VACANCY, "vacancy"),
            m("development_completions", r"Development completions reached {n} million sq ft in Q2 2026, with 11 new developments and major refurbishments delivered across Central London\.", "Central London", "million sq ft", COMPLETIONS, "pipeline"),
            m("under_construction", r"Overall space under construction stood at {n} million sq ft at the end of H1", "Central London", "million sq ft", "Office space under construction at reference-period end; includes pre-let space", "pipeline"),
            m("prime_rent", r"West End average prime rent at £{n} per sq ft was up 2\.3% on H1 2025", "West End", "GBP per sq ft", PRIME, "prime_rent", H1),
            m("grade_a_rent", r"City average Grade A rent reaching £{n} per sq ft, up 7% on H1 2025", "City", "GBP per sq ft", GRADE_A, "grade_a_rent", H1),
            m("grade_a_rent", r"West End, the average Grade A rent stood at £{n} per sq ft at the end of H1", "West End", "GBP per sq ft", GRADE_A, "grade_a_rent", H1),
        ),
        (
            ("prime_rent", r"City prime rent stood at £116\.73 per sq ft, up 19% year-on-year, having reached a new record in Q1\."),
            ("occupier_demand", r"Notable Q2 transactions to AI occupiers include[^.]{1,550}\."),
            Passage("pipeline", r"However, development starts fell sharply in H1 2026, with just 1\.5 million sq ft commencing,[^.]{1,500}\. Only one new-build scheme broke ground during the period[^.]{1,450}\. The remaining 14 starts were refurbishments\.", "Central London", "Central London", facets=("new_development", "refurbishment")),
            Passage("pipeline", r"Overall, based on our analysis of the development pipeline \(with a view on schemes with a realistic prospect of delivery\), development completions are expected to reach 22 million sq ft by the end of 2029, with 148 schemes scheduled for completion over this period\. However, 28% of this future supply has yet to commence construction[^.]{1,450}\.", "Central London", "Central London", kind="forecast", qualifications=(ForecastRiskSpec(r"development completions are expected to reach (?P<quantity>\d+(?:\.\d+)?) million sq ft by the end of (?P<year>\d{4}), with (?P<schemes>\d+) schemes scheduled for completion over this period\.", r"However, (?P<share>\d+(?:\.\d+)?)% of this future supply has yet to commence construction and therefore remains vulnerable to further delays or potential withdrawal as viability pressures persist\.", "million sq ft"),)),
            Passage("occupier_demand", r"Our rental analysis indicates buildings with a BREEAM certification[^.]{1,300}\. In H1, average rents[^.]{1,300}\.", "Central London", "Central London", facets=("esg",)),
        ),
        ("Q2 City prime rent lacks an explicit averaging window and remains qualitative only. West End prime and Grade A rents use the stated H1 period.", "Q2 West End vacancy has only a change in bps, so no absolute observation is derived.", "Reported supply totals 19.1m and 18.7m imply a 0.4m change, while Q2 narrative says 0.5m; retain rounded source totals and flag the discrepancy.", "Projected completions are qualitative source forecasts, not current availability.", "No normalized Canary Wharf, Midtown and Fringe or wider-London series is established."), version="6",
    ),
    "savills-central-london-june-2026": MonitoredSource(
        "savills-central-london-june-2026", "https://www.savills.com/research_articles/255800/392507-0",
        "Central London Office Market Watch 30 June 2026 Article", date(2026, 6, 30), MAY,
        (
            m("take_up", r"Take-up across Central London reached {n} sq ft in May across \d+ transactions", "Central London", "sq ft", TAKEUP, "take-up"),
            m("prime_rent", r"average achieved City prime rent for the year to date stands at £{n} per sq ft", "City", "GBP per sq ft", PRIME, "prime_rent", (date(2026, 1, 1), date(2026, 5, 31))),
            m("prime_rent", r"average achieved West End prime rent reached £{n} per sq ft", "West End", "GBP per sq ft", PRIME, "prime_rent", (date(2026, 1, 1), date(2026, 5, 31))),
        ),
        (("grade_a_rent", r"Average City Grade A rent stood at £79\.65 per sq ft at the end of May, up 13% on the same period in 2025, while the average West End Grade A rent reached £103\.91 per sq ft, up 3% year-on-year\."), ("occupier_demand", r"This transaction continued the trend of growing demand from the AI sub-sector[^.]{1,650}\."),),
        ("Published 30 June but reports May activity, not June activity.", "Rent averages are retained as January-May cumulative measures, not a monthly trend.", "Monthly supply sentence links April and May ambiguously; excluded from normalization. Grade A averaging windows are unspecified and remain qualitative only.", "Central London does not establish whole-London coverage."),
    ),
}

Q3_2025 = (date(2025, 7, 1), date(2025, 9, 30))
Q4_2025 = (date(2025, 10, 1), date(2025, 12, 31))
BNP_TAKEUP = "BNP Paribas Real Estate reported Central London office space leased during the reference period"
BNP_VACANCY = "BNP Paribas Real Estate reported office vacancy rate at reference-period end; comparability of publisher definitions and boundaries is not established"
BNP_SECTOR_SHARE = "BNP Paribas Real Estate reported occupier-sector share of Q1 2026 Central London take-up by area; selected sectors only"

MONITORED_SOURCES.update({
    "deutsche-bank-canary-wharf-2026": MonitoredSource(
        "deutsche-bank-canary-wharf-2026", "https://www.db.com/news/detail/20260819-deutsche-bank-to-take-space-at-yy-london-canary-wharf-from-early-2028?language_id=1",
        "Media Release London, August 20, 2026 Deutsche Bank to take space at YY London, Canary Wharf, from early 2028", date(2026, 8, 20), None, (),
        (
            Passage("news", r"Deutsche Bank today announces that it has secured office space for its infrastructure functions at YY London in Canary Wharf, following an extensive review of options ahead of the expiry of its current lease at 10 Upper Bank Street in July 2028\.", "Canary Wharf", "Canary Wharf", "Occupier's original announcement of secured space, with planned occupation in early 2028. Not occupied space in August 2026; no floor area is stated.", facets=('news',)),
            Passage("occupier_demand", r"Building: Fully electric, targeting net-zero-in-operation Workplace amenities: Modern working areas, in-house cafe options, commuter facilities, rooftop gardens and terraces", "Canary Wharf", "Canary Wharf", "Source describes target sustainability performance and amenities, not independently verified net-zero achievement or quantified rental premium.", facets=('esg',)),
            Passage("news", r"Deutsche Bank to take space at YY London, Canary Wharf, from early 2028", "Canary Wharf", "Canary Wharf", "Fact of the occupier's announced future plan; the headline does not establish observed occupation in 2026.", kind="fact", facets=('news',)),
            Passage("news", r"Planned occupation: Early 2028", "Canary Wharf", "Canary Wharf", "Fact of the announced occupation window; a future plan, not an exact commencement date or completed event.", kind="fact", facets=('news',)),
        ),
        ("Original occupier announcement gives planned occupation in early 2028, not an exact commencement date or observed market take-up series.", "The URL contains 19 August but the visible release is dated 20 August 2026; publication uses the article's date.", "One occupier event does not establish Canary Wharf-wide rent, vacancy or take-up."),
        version="3", publisher="Deutsche Bank", source_kind="primary_news", topics=("news", "take_up", "occupier_demand"), geographies=("Canary Wharf",),
        discovery_query="Deutsche Bank Canary Wharf YY London office latest news",
        discovery_urls=("https://www.db.com/news/", "https://cwg.com/media/press-releases/"),
    ),
    "savills-central-london-q3-2025": MonitoredSource(
        "savills-central-london-q3-2025", "https://www.savills.com/research_articles/255800/381989-0",
        "Central London Office Market Watch Q3 2025 27 October 2025 Article", date(2025, 10, 27), Q3_2025,
        (
            m("take_up", r"Leasing activity slowed in Q3, with take-up reaching {n} million sq ft across 167 transactions\.", "Central London", "million sq ft", TAKEUP, "take-up"),
            m("vacancy_rate", r"overall supply standing at 20\.2 million sq ft, equating to a vacancy rate of {n}%", "Central London", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"The City vacancy rate stood at {n}%, down 10 bps", "City", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"West End vacancy rate rose by 20 bps over the quarter to {n}%", "West End", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"City Fringe vacancy rate, which currently stands at {n}%", "Midtown and Fringe", "%", VACANCY, "vacancy", native_geography="City Fringe"),
        ),
        (Passage("occupier_demand", r"The preference for newer quality offices is reflected by the fact that recently redeveloped or comprehensively refurbished office space accounted for 70% of space acquired so far this year, with BREEAM-rated Excellent and Outstanding offices representing 61% of space acquired\.", "Central London", "Central London", "These are January-September cumulative shares, not Q3-only shares or an AI-specific share.", facets=('flight_to_quality', 'esg')),),
        ("Quarterly vacancy and take-up history is compatible only within this publisher's definitions.", "City Fringe maps to Midtown and Fringe but does not establish Midtown evidence.", "Q3 rent averages have uncertain cumulative windows and are not normalized."),
        geographies=("Central London", "City", "West End", "Midtown and Fringe"),
    ),
    "savills-central-london-q4-2025": MonitoredSource(
        "savills-central-london-q4-2025", "https://www.savills.com/research_articles/255800/386949-0",
        "Central London Office Market Watch Q4 2025 30 January 2026 Article", date(2026, 1, 30), Q4_2025,
        (
            m("take_up", r"Central London take-up totalled {n} million sq ft in Q4, with 188 transactions completed\.", "Central London", "million sq ft", TAKEUP, "take-up"),
            m("vacancy_rate", r"This equates to a vacancy rate of {n}%, which represents a 40bps fall on the previous quarter", "Central London", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"Both the City and West End recorded a decline in supply over the quarter, with vacancy rates reaching {n}% and 7\.9%, respectively\.", "City", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"Both the City and West End recorded a decline in supply over the quarter, with vacancy rates reaching 7\.0% and {n}%, respectively\.", "West End", "%", VACANCY, "vacancy"),
            m("vacancy_rate", r"City Fringe vacancy rate, which stood at {n}%", "Midtown and Fringe", "%", VACANCY, "vacancy", native_geography="City Fringe"),
            m("vacancy_rate", r"West End Fringe sub-markets such as Hammersmith and VNEB, where Q4 vacancy stood at {n}% and 18%, respectively\.", "Wider London", "%", VACANCY, "vacancy", native_geography="Hammersmith"),
            m("vacancy_rate", r"West End Fringe sub-markets such as Hammersmith and VNEB, where Q4 vacancy stood at 26% and {n}%, respectively\.", "Wider London", "%", VACANCY, "vacancy", native_geography="VNEB"),
        ),
        (Passage("occupier_demand", r"Space let in BREEAM Outstanding or Excellent office buildings represented 64% of space acquired during 2025, up 5% on 2024\.", "Central London", "Central London", "Annual 2025 share, not Q4-only. Association does not establish a causal ESG rent premium.", facets=('esg',)),),
        ("Research article vacancy is 7.4%; a separate Savills press release reports 7.5%. This contract uses the research article and does not reconcile the discrepancy.", "Hammersmith and VNEB are retained separately with source-native labels inside Wider London; these locations do not establish exhaustive wider-London coverage.", "City Fringe does not establish Midtown coverage. Annual rent and occupier shares are not quarterly increments."),
        geographies=("Central London", "City", "West End", "Midtown and Fringe", "Wider London"),
    ),
    "bnp-central-london-q1-2026": MonitoredSource(
        "bnp-central-london-q1-2026", "https://www.realestate.bnpparibas.co.uk/insights/central-london-office-market-update-q1-2026",
        "Research 18.05.2026 Central London Office Market Update Q1 2026", date(2026, 5, 18), Q1,
        (
            m("take_up", r"Total take-up reached {n}m sq ft", "Central London", "million sq ft", BNP_TAKEUP, "take-up"),
            m("take_up", r"The City recorded a measured opening to the year, with take-up totalling {n}m sq ft", "City", "million sq ft", BNP_TAKEUP, "take-up"),
            m("take_up", r"West End experienced a stronger start to the year, with take-up rising to {n}m sq ft", "West End", "million sq ft", BNP_TAKEUP, "take-up"),
            m("vacancy_rate", r"overall Central London vacancy rate reduced to {n}%", "Central London", "%", BNP_VACANCY, "vacancy"),
            m("vacancy_rate", r"The West End retained the lowest vacancy profile across Central London, declining further to {n}% in Q1 2026", "West End", "%", BNP_VACANCY, "vacancy"),
            m("vacancy_rate", r"The City followed a similar trajectory, with vacancy reducing to {n}%", "City", "%", BNP_VACANCY, "vacancy"),
            m("occupier_take_up_share", r"Media & Tech emerged as the most active occupier group during Q1 2026, accounting for {n}% of total take-up", "Central London", "%", BNP_SECTOR_SHARE, "occupier_demand", dimension="occupier_sector", segment="Media & Tech", partition="bnp-q1-2026-central-london-occupier-sector-share"),
            m("occupier_take_up_share", r"The Services sector ranked second with a {n}% share of activity", "Central London", "%", BNP_SECTOR_SHARE, "occupier_demand", dimension="occupier_sector", segment="Services", partition="bnp-q1-2026-central-london-occupier-sector-share"),
            m("occupier_take_up_share", r"Banking & Finance represented {n}% of quarterly take-up", "Central London", "%", BNP_SECTOR_SHARE, "occupier_demand", dimension="occupier_sector", segment="Banking & Finance", partition="bnp-q1-2026-central-london-occupier-sector-share"),
            m("take_up_grade_share", r"Grade A space accounting for {n}% of all take-up during Q1 2026", "Central London", "%", "BNP Paribas Real Estate Grade A share of Q1 2026 Central London take-up by area; not a share of stock", "occupier_demand", dimension="office_grade", segment="Grade A"),
            m("supply_grade_share", r"Grade B stock continues to dominate overall availability, accounting for {n}% of total supply across Central London", "Central London", "%", "BNP Paribas Real Estate Grade B share of Central London office supply at Q1 end; not vacancy or take-up share", "availability", dimension="office_grade", segment="Grade B"),
        ), (),
        ("BNP and Savills are separate publisher series. Retained excerpts do not establish comparable definitions or boundaries or reconcile different figures; do not splice their histories.", "Three occupier sectors are a partial partition totalling 60.3%; no Other share is inferred.", "Grade A take-up and Grade B supply have different denominators and cannot form one proportion chart. Prime rent averaging windows are not established and are excluded."),
        version="3", publisher="BNP Paribas Real Estate", geographies=("Central London", "City", "West End"),
        topics=("vacancy_availability", "take_up", "submarket_dynamics", "occupier_demand"),
        discovery_query="BNP Paribas Real Estate Central London Office Market Update latest quarter",
        discovery_urls=("https://www.realestate.bnpparibas.co.uk/insights/research",),
    ),
    "boe-september-2026": MonitoredSource(
        "boe-september-2026", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/september-2026",
        "Published on 17 September 2026 Monetary Policy Summary, September 2026", date(2026, 9, 17), (date(2026, 9, 16), date(2026, 9, 16)),
        (m("bank_rate", r"At its meeting ending on 16 September 2026, the Monetary Policy Committee \(MPC\) voted by a majority of 6–3 to maintain Bank Rate at {n}%\.", "Unmapped", "%", "Bank of England MPC policy Bank Rate at the dated decision; UK-wide policy rate, not London office financing cost", "macro", native_geography="United Kingdom"),),
        (
            Passage("macro", r"UK CPI inflation increased to 3\.1% in August", "United Kingdom", "Unmapped", "Observed August UK CPI inflation; not a London-specific rate or rent forecast.", kind="fact"),
            Passage("macro", r"and is likely to rise further over coming quarters\.", "United Kingdom", "Unmapped", "The antecedent is UK CPI inflation; this fragment is the Bank's future inflation outlook, not observed inflation.", kind="forecast"),
        ),
        ("Official UK policy influences London offices but is not a submarket observation.", "Retained HTML is truncated after 18,000 characters; the reviewed opening policy statement is within that bound. Later minutes are not approved by this contract."),
        version="3", publisher="Bank of England", source_kind="official_statistics", topics=("macro",), geographies=("Unmapped",),
        discovery_query="Bank of England latest Monetary Policy Summary and Minutes Bank Rate",
        discovery_urls=("https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes",),
    ),
    "ons-hybrid-work-2025": MonitoredSource(
        "ons-hybrid-work-2025", "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/articles/whohasaccesstohybridworkingreatbritain/2025-06-11",
        "Release date: 11 June 2025", date(2025, 6, 11), (date(2025, 1, 8), date(2025, 3, 30)),
        (m("hybrid_work_share", r"More than a quarter of workers in Great Britain \({n}%\) were hybrid workers between 8 January and 30 March 2025\.", "Unmapped", "%", "ONS Opinions and Lifestyle Survey working adults reporting home and workplace working in the previous seven days; Great Britain, not London-specific", "occupier_demand", native_geography="Great Britain"),), (),
        ("Survey working adults are not office-only workers and cannot establish London office attendance or occupier floor-area requirements.", "Visible release date is 11 June 2025; HTML datePublished metadata reports 10 June. The contract retains the visible official release date and this discrepancy.", "Historical January-March 2025 observation remains stale context, not a 2026 change. Retained text is truncated; the reviewed survey phrase precedes truncation."),
        publisher="Office for National Statistics", source_kind="official_statistics", topics=("occupier_demand",), geographies=("Unmapped",),
        discovery_query="ONS hybrid working Great Britain latest working arrangements London",
        discovery_urls=("https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork",), accepted_metadata_dates=(date(2025, 6, 10),),
    ),
    "gla-london-economy": MonitoredSource(
        "gla-london-economy", "https://apps.london.gov.uk/state-of-london/report/economy",
        "London generates roughly a quarter of the UK’s economic output", None, None, (),
        (
            Passage("macro", r"With the official London unemployment rate having hit 7\.4% in March 2026, this suggests that weaker demand in the labour market is the key contributor recently to involuntary worklessness in London\.", "London", "Whole London", "GLA commentary attributes the unemployment rate to official statistics; exact three-month survey window is unspecified, so no normalized observation is created. The weaker-demand explanation is attributed interpretation.", kind="interpretation", facets=('employment',)),
            Passage("macro", r"The most recent ONS data release showed London’s real GVA grew by only 0\.3% in 2023 \(compared to 0\.4% for the UK as a whole\)\.", "London", "Whole London", "GLA reports an observed 2023 ONS release; this is not a new primary ONS release or current growth measurement.", kind="fact", facets=('economy',)),
            Passage("macro", r"Estimates by GLA Economics suggest a modest strengthening of economic growth in London over the last two years, with real GVA expected to have risen by 1\.1% in 2024 and 1\.9% in 2025\.", "London", "Whole London", "2024 and 2025 are GLA estimates, not observed official outturn. The existing source-forecast kind preserves this estimated status.", kind="forecast", facets=('economy',)),
        ),
        ("Dynamic GLA chapter has no confirmed publication date; retrieval timestamp is not publication date.", "Economic and employment context is whole-London but cannot establish office rent, vacancy or take-up coverage. HTML charts show loading placeholders; no plotted values are extracted.", "The provider retains the first 18,000 characters only; reviewed passages are inside that bound."),
        version="3", publisher="Greater London Authority", source_kind="official_analysis", topics=("macro",), geographies=("Whole London",),
        discovery_query="GLA Economics London labour market unemployment employment latest economy",
        discovery_urls=("https://data.london.gov.uk/",),
    ),
    "lendlease-stratford-cross-2026": MonitoredSource(
        "lendlease-stratford-cross-2026", "https://www.lendlease.com/media-centre/media-releases/plans-submitted-final-phase-stratford-cross/",
        "Plans submitted for final phase of Stratford Cross 20 Aug 2026", date(2026, 8, 20), (date(2026, 8, 20), date(2026, 8, 20)), (),
        (Passage("news", r"Plans have been submitted to the London Borough of Newham for the final phase of development at Stratford Cross\. The proposals would deliver approximately 2,000 new homes including affordable housing, alongside a new public park, active ground-floor retail space and enhanced walking and cycling routes linking Stratford station, Queen Elizabeth Olympic Park and the surrounding neighbourhoods\.", "Stratford", "Wider London", "Submitted residential-led proposal, not consented development or completed office supply; relevant competing land-use news around an established commercial district.", facets=('news',)),),
        ("Developer announcement describes a planning submission and proposed homes; neither office completions nor planning consent can be inferred.", "One Stratford news item does not establish a wider-London office market series."),
        publisher="Lendlease", source_kind="primary_news", topics=("news", "supply_pipeline"), geographies=("Wider London",),
        discovery_query="Stratford Cross Lendlease office development planning latest",
        discovery_urls=("https://www.lendlease.com/media-centre/media-releases/",),
    ),
    "lendlease-turing-stratford-2025": MonitoredSource(
        "lendlease-turing-stratford-2025", "https://www.lendlease.com/media-centre/media-releases/lendlease-unveils-new-east-london-landmark/",
        "Lendlease unveils new East London landmark 8 Jan 2025", date(2025, 1, 8), None, (),
        (
            Passage("pipeline", r"Located at Stratford Cross, the Turing Building includes 350,000 square feet of sustainable Grade A office space and was designed by architects Arney Fender Katsalidis\. It’s the first building to be delivered as part of Lendlease’s joint venture with Canada Pension Plan Investment Board \(CPP Investments\) at Stratford Cross on the edge of the Queen Elizabeth Olympic Park\.", "Stratford", "Wider London", "Developer describes a delivered individual building. The exact completion date and currently unlet area are not established; 350,000 sq ft is building office area, not market take-up or current availability.", facets=('new_development', 'esg')),
            Passage("occupier_demand", r"The office building has dual consent for both office and educational use, making it a flexible and unique offering for businesses and educational institutions\.", "Stratford", "Wider London", "Dual-use consent and developer positioning are qualitative supply and demand context; no occupancy or rent evidence is stated."),
        ),
        ("Historical developer announcement from January 2025, not a newly completed building in October 2026.", "Office area is an individual building description and cannot establish market-wide available supply. Exact delivery date and current letting status require further research."),
        publisher="Lendlease", source_kind="primary_news", topics=("news", "supply_pipeline", "occupier_demand"), geographies=("Wider London",),
        discovery_query="Turing Building Stratford Cross office leasing availability latest Lendlease",
        discovery_urls=("https://www.lendlease.com/media-centre/",),
    ),
    "cityam-london-polarisation-2026": MonitoredSource(
        "cityam-london-polarisation-2026", "https://www.cityam.com/demand-soars-for-central-london-offices-while-outskirts-snubbed/",
        "Thursday 09 April 2026 2:35 pm Demand soars for central London offices while outskirts snubbed", date(2026, 4, 9), None, (),
        (Passage("news", r"Hammersmith, the outer edge of London’s East End and Hounslow are among the worst-performing areas for office space in the capital, CoStar found\.", "Hammersmith, East End and Hounslow", "Wider London", "Secondary City AM reporting attributed to CoStar, not independently verified primary market data. No quantitative observations or publisher-comparable series are created.", facets=('news',)),),
        ("Secondary reporting, explicitly attributed to CoStar. Core/fringe boundaries and reference periods are not established; its rates must not be compared with Savills or BNP.", "Retained excerpt is 25 words; full article remains at the source URL. Wider London coverage is qualitative and selective."),
        publisher="City AM", source_kind="secondary_news", topics=("news", "vacancy_availability", "submarket_dynamics"), geographies=("Wider London",),
        discovery_query="City AM London offices leasing Canary Wharf Hammersmith property latest",
        discovery_urls=("https://www.cityam.com/",),
    ),
})


MONITORED_SOURCES["ons-regional-labour-september-2026"] = MonitoredSource(
    "ons-regional-labour-september-2026",
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/regionallabourmarket/september2026",
    "Labour market in the regions of the UK: September 2026", date(2026, 9, 15), (date(2026, 5, 1), date(2026, 7, 31)),
    (
        m("employment_rate", r"London {n} -0\.4 6\.8 0\.1 20\.5 0\.2", "Whole London", "%", "ONS Labour Force Survey seasonally adjusted employment rate for London residents aged 16 to 64; not London workplace or office employment", "employment", native_geography="London"),
        m("unemployment_rate", r"London 73\.9 -0\.4 {n} 0\.1 20\.5 0\.2", "Whole London", "%", "ONS Labour Force Survey seasonally adjusted unemployment rate for economically active London residents aged 16 and over; not office-worker unemployment", "employment", native_geography="London"),
        m("inactivity_rate", r"London 73\.9 -0\.4 6\.8 0\.1 {n} 0\.2", "Whole London", "%", "ONS Labour Force Survey seasonally adjusted economic inactivity rate for London residents aged 16 to 64; not office-worker inactivity", "employment", native_geography="London"),
    ), (
        Passage("labour_definitions", r"Employment rate \(\%\) aged 16 to 64 years \[Note 2\] Change on February to April 2026 Unemployment rate \(\%\) aged 16 years and over \[Note 3\] Change on February to April 2026 Inactivity rate \(\%\) aged 16 to 64 years \[Note 4\] Change on February to April 2026", "United Kingdom", "Unmapped", "Exact regional table headers define separate age cohorts; use with the London row, not as an employment-level finding. Unemployment uses the economically active population; employment and inactivity use the corresponding resident population.", kind="fact"),
        Passage("labour_definitions", r"The unemployment rate is not the proportion of the total population who are unemployed\. It is the proportion of the economically active population \(those in work plus those seeking and available to work\) who are unemployed\.", "United Kingdom", "Unmapped", "Exact population denominator from the source glossary; combine with the separate age header and London row.", kind="fact"),
        Passage("labour_definitions", r"LFS, APS and Real Time Information \(RTI\) estimates are based on the region where someone lives, as opposed to the region where they may happen to work, which may be different\.", "United Kingdom", "Unmapped", "Residence-based regional definition, not London workplace jobs or office workers; contextual support for the London row.", kind="fact"),
    ),
    (
        "Residence-based London LFS estimates, not London workplace jobs, office workers, office attendance or occupier floor-space demand.",
        "May-July three-month levels. Reported quarterly changes in the table are not weekly changes; this contract normalizes levels only.",
        "Sampling volatility remains, especially granular estimates; short-term movements may be within sampling variability. The source advises considering longer-term patterns and other indicators.",
        "Visible release date is 15 September 2026; metadata reports 14 September UTC. The reviewed contract retains the visible release date.",
    ),
    version="4", publisher="Office for National Statistics", source_kind="official_statistics", topics=("macro", "employment"), geographies=("Whole London",),
    discovery_query="ONS London regional labour market employment unemployment latest",
    discovery_urls=("https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork",),
    accepted_metadata_dates=(date(2026, 9, 14),),
    required_phrases=(
        "Release date: 15 September 2026",
        "Table 1: Summary of latest headline estimates and quarterly changes, for regions of the UK, seasonally adjusted, May to July 2026",
        "Employment rate (%) aged 16 to 64 years [Note 2] Change on February to April 2026 Unemployment rate (%) aged 16 years and over [Note 3] Change on February to April 2026 Inactivity rate (%) aged 16 to 64 years [Note 4] Change on February to April 2026",
        "Some volatility remains, particularly for estimates for mid-2023 and throughout 2024, and for granular breakdowns, where sample sizes are smaller.",
        "Caution should be taken when drawing conclusions from short-term changes, and we advise users to focus on long-term movements in the data.",
        "LFS, APS and Real Time Information (RTI) estimates are based on the region where someone lives, as opposed to the region where they may happen to work, which may be different.",
    ),
)

MONTHLY_ACQUIRED = "Savills reported Central London office space acquired during the month, including owner-occupier acquisitions where counted; not a pure leased-space series"
MONTHLY_SCOPE = "exploring insight from the City and West End office occupational markets"

MONITORED_SOURCES.update({
    "savills-central-london-august-2026": MonitoredSource(
        "savills-central-london-august-2026", "https://www.savills.co.uk/research_articles/229130/394196-0/central-london-office-market-watch---august-2026",
        "Central London Office Market Watch 28 August 2026 Article", date(2026, 8, 28), (date(2026, 7, 1), date(2026, 7, 31)),
        (m("space_acquired", r"48 transactions totalling {n} sq ft took place across Central London, with the West End accounting for 53% of space acquired\.", "Central London", "sq ft", MONTHLY_ACQUIRED, "take-up"),), (),
        ("Published 28 August, reporting July monthly activity, not weekly observations.", "The July total includes an owner-occupier freehold acquisition; acquired space is not a pure leased-space series.", "The publisher's reported August 35% month-on-month change needs reconciliation against the separately dated July and August totals. No revised value or revision history is inferred.", "City and West End scope does not establish whole-London quantitative coverage."),
        version="1", topics=("take_up", "submarket_dynamics", "occupier_demand"), geographies=("Central London",),
        discovery_query="Savills latest Central London Office Market Watch monthly activity",
        discovery_urls=("https://www.savills.co.uk/research/",),
        required_phrases=(MONTHLY_SCOPE, "leasing activity moderated in July"),
    ),
    "savills-central-london-september-2026": MonitoredSource(
        "savills-central-london-september-2026", "https://www.savills.co.uk/research_articles/229130/394968-0",
        "Central London Office Market Watch 30 September 2026 Article", date(2026, 9, 30), (date(2026, 8, 1), date(2026, 8, 31)),
        (m("space_acquired", r"A total of {n} sq ft was acquired across Central London, down 35% month on month \(MoM\) but 10% above August 2025, with the City accounting for 67% of space acquired\.", "Central London", "sq ft", MONTHLY_ACQUIRED, "take-up", qualifications=(ReportedChangeSpec(r"down (?P<change>35)% month on month \(MoM\)", (date(2026, 7, 1), date(2026, 7, 31)), (date(2026, 8, 1), date(2026, 8, 31)), "%", basis="relative_percent", requires_comparison=True),)),), (),
        ("Published 30 September, reporting August monthly activity, not September or weekly observations.", "The publisher's reported August 35% month-on-month change needs reconciliation against the separately dated July and August totals. No revised value or revision history is inferred.", "City and West End scope does not establish whole-London quantitative coverage."),
        version="2", topics=("take_up", "submarket_dynamics", "occupier_demand"), geographies=("Central London",),
        discovery_query="Savills latest Central London Office Market Watch monthly activity",
        discovery_urls=("https://www.savills.co.uk/research/",),
        required_phrases=(MONTHLY_SCOPE, "leasing activity was quiet in August"),
    ),
    "london-fire-lambeth-hq-october-2026": MonitoredSource(
        "london-fire-lambeth-hq-october-2026", "https://www.london-fire.gov.uk/news/2026-news/october/london-fire-brigade-set-for-return-to-its-historic-hq-in-lambeth/",
        "London Fire Brigade set for return to its historic HQ in Lambeth 06/10/2026 12:00", date(2026, 10, 6), None, (),
        (
            Passage("news", r"London Fire Brigade is set to return to its historic headquarters on Albert Embankment after receiving planning permission for plans at the site from Lambeth Council\. This will see its historic site refurbished to provide a fire station, office space and community spaces\.", "Lambeth", "Wider London", "Fact of the announced planning permission and future plan to return. The permission decision date is not established; publication is not the event date. Not a completed letting, occupation, refurbishment, new construction or market-wide supply change; no area is stated.", kind="fact", facets=('refurbishment', 'news')),
        ),
        ("Original public-body announcement published 6 October; planned HQ return and refurbishment remain future events.", "One Lambeth occupier event does not establish wider-London rent, vacancy, take-up or available supply."),
        version="1", publisher="London Fire Brigade", source_kind="primary_news", topics=("news", "occupier_demand", "supply_pipeline"), geographies=("Wider London",),
        discovery_query="London Fire Brigade Lambeth headquarters refurbishment planning news",
        discovery_urls=("https://www.london-fire.gov.uk/news/",),
    ),
    "knight-frank-london-commentary-october-2026": MonitoredSource(
        "knight-frank-london-commentary-october-2026", "https://www.knightfrank.co.uk/research/podcast-pages/2026/10/the-case-for-real-estate-in-a-changed-market",
        "The case for real estate in a changed market", date(2026, 10, 7), None, (),
        (Passage("occupier_demand", r"In London offices, development activity is not keeping pace with occupier demand for the type of space now required, a dynamic driven partly by viability pressures and partly by the same macroeconomic caution weighing on investors\.", "London offices", "Whole London", "Attributed Knight Frank commentary by William Matthews, in an episode with Shabab Qadar, Head of London Offices Research. Interpretation, not a quantified observed London-office indicator. No measurement reference period or independently established causal explanation."),),
        ("Published 7 October. Research commentary has no quantified measurement or observation reference period and creates no normalized series.", "Broad London commentary cannot establish complete submarket coverage or corroborate a numerical Savills observation."),
        version="1", publisher="Knight Frank", source_kind="primary_research_commentary", topics=("news", "occupier_demand", "supply_pipeline"), geographies=("Whole London",),
        discovery_query="Knight Frank London office development demand latest research commentary",
        discovery_urls=("https://www.knightfrank.co.uk/research/",),
        required_phrases=("07 October 2026", "Shabab Qadar, Head of London Offices Research", "William Matthews"),
    ),
    "semafor-canary-wharf-october-2026": MonitoredSource(
        "semafor-canary-wharf-october-2026", "https://www.semafor.com/article/10/09/2026/the-ceo-who-put-canary-wharf-back-on-the-map",
        "The CEO who put Canary Wharf back on the map", date(2026, 10, 9), None, (),
        (
            Passage("news", r"Fast forward to 2026, and Canary Wharf Group is reporting the highest rental rates in its history\. Office occupancy now exceeds 92%", "Canary Wharf", "Canary Wharf", "Semafor reports Canary Wharf Group's company/estate claims. The rent definition, asking versus achieved basis and reference period are unspecified. Reported occupancy is not office-worker attendance or a market-wide vacancy measurement.", kind="fact"),
            Passage("news", r"visitor numbers, which fell from 54 million a year to just 20 million during the pandemic, are on track to exceed last year’s record of 76 million\.", "Canary Wharf", "Canary Wharf", "Semafor reports a Canary Wharf visitor forecast: 76 million is the previous year's record, not an achieved current-year count. Visitor footfall is not office attendance, occupied office area or market take-up.", kind="forecast"),
            Passage("hybrid", r"Staff are still pushing back against their employers’ efforts to get them back to their desks for three, four, or five days a week, and Canary Wharf’s tenants have differing policies on how often they expect people in their offices\.", "Canary Wharf", "Canary Wharf", "Semafor's qualitative reporting of staff resistance and differing Canary Wharf tenant attendance policies; no measured attendance rate or market-wide policy is established.", kind="fact"),
            Passage("hybrid", r"But they increasingly share one view, Khan says: “You’ve got to have the best space to compete for that talent\.”", "Canary Wharf", "Canary Wharf", "Semafor attributes this tenant talent/space opinion to Canary Wharf Group CEO Shobi Khan. It is commentary, not measured attendance, a causal rent premium or independently verified occupier demand.", kind="interpretation"),
        ),
        ("Secondary Semafor reporting of Canary Wharf Group claims, not independently verified market observations.", "Company/estate scope and unspecified rent definition/reference period prevent normalization. Visitor forecasts and reported office occupancy do not measure office attendance."),
        version="2", publisher="Semafor", source_kind="secondary_news", topics=("news", "occupier_demand"), geographies=("Canary Wharf",),
        discovery_query="Semafor Canary Wharf Group rental rates occupancy visitor forecast",
        discovery_urls=("https://www.semafor.com/",), required_phrases=("Oct 9, 2026", "Canary Wharf Group"),
    ),
    "cyprus-mail-uk-defence-october-2026": MonitoredSource(
        "cyprus-mail-uk-defence-october-2026", "https://cyprus-mail.com/2026/10/05/tech-driven-defence-boom-sparks-demand-for-uk-offices-and-research-labs",
        "Tech-driven defence boom sparks demand for UK offices and research labs |  Cyprus Mail", date(2026, 10, 5), None, (),
        (Passage("occupier_demand", r"In Britain, up to 3 million square metres of industrial and logistics demand will be needed, \u200bplus nearly 250,000 square metres of additional office and research and development space, Savills forecasts\.", "United Kingdom", "Unmapped", "Cyprus Mail's Reuters-sourced report attributes these defence-related UK demand forecasts to Savills. Nearly 250,000 square metres covers office and research/development space; up to 3 million square metres separately covers industrial/logistics demand. No UK horizon is explicitly stated: do not insert 2040 or transfer the preceding European seven-year horizon. UK context is not a London office measurement.", kind="forecast"),),
        ("Reuters-sourced secondary reporting, not original Savills research or an observed letting.", "Defence-related UK forecasts retain nearly/up-to bounds and distinct office/R&D versus industrial/logistics scope. The UK horizon is unspecified; the European seven-year estimate does not supply it."),
        version="1", publisher="Cyprus Mail (Reuters-sourced)", source_kind="secondary_news", topics=("news", "occupier_demand"), geographies=("Unmapped",),
        discovery_query="Reuters Britain defence demand offices research development Savills forecast",
        discovery_urls=("https://cyprus-mail.com/",),
        required_phrases=("real estate companies have told Reuters.", "A boom in defence technology", "CANARY WHARF ESTATE SEEN AS SECURE"),
        allow_document_title=True,
    ),
})


@dataclass(frozen=True)
class ParsedSource:
    document: Document
    evidence: list[Evidence]
    observations: list[Observation]
    caveats: tuple[str, ...]


def evidence_qualifications(source: MonitoredSource, metric: str, excerpt: str, period=None) -> str:
    notes = []
    if metric == "reported_supply":
        notes.append("Source-reported supply has no precise equivalence to marketed availability established by this contract.")
        if source.id == "savills-central-london-q2-2026":
            notes.append("Reported supply totals 19.1m and 18.7m imply a 0.4m change, while Q2 narrative says 0.5m; retain rounded source totals and flag the discrepancy.")
    if source.id == "savills-central-london-q2-2026" and "City prime rent stood" in excerpt:
        notes.append("Q2 City prime rent lacks an explicit averaging window and remains qualitative only; it cannot be compared with the H1 West End average.")
    if metric in ("prime_rent", "grade_a_rent") and period:
        if period == H1:
            notes.append("H1 January-June average; not comparable with a quarterly or January-May average.")
        elif period == (date(2026, 1, 1), date(2026, 5, 31)):
            notes.append("January-May cumulative average; not monthly or quarterly increments.")
    if source.id == "savills-central-london-june-2026":
        if metric == "grade_a_rent" and period is None:
            notes.append("Grade A averaging windows are unspecified and remain qualitative only.")
        elif metric == "take_up":
            notes.append("Published 30 June but reports May take-up, not June activity.")
    if "development completions are expected to reach" in excerpt:
        notes.append("Projected completions are qualitative source forecasts, not current availability.")
    return " Source qualifications. " + " ".join(notes) if notes else ""


def parse_monitored_source(source_id: str, page: FetchedPage) -> ParsedSource:
    source = MONITORED_SOURCES.get(source_id)
    if source is None:
        raise ValueError("unknown monitored source ID")
    doc = page.document
    if doc.mode != "live" or doc.synthetic or str(doc.url).rstrip("/") != source.url.rstrip("/") or doc.publisher != urlsplit(source.url).hostname:
        raise ValueError("document does not match the reviewed live source identity")
    accepted_dates = (source.published_at, *source.accepted_metadata_dates)
    if (source.header not in page.text and not (source.allow_document_title and doc.title == source.header)) or (doc.published_at is not None and doc.published_at not in accepted_dates):
        raise ValueError("reviewed article title and publication date did not match")
    if any(phrase not in page.text for phrase in source.required_phrases):
        raise ValueError("reviewed source scope, reference period or attribution did not match")
    matched_measurements = []
    for measurement in source.measurements:
        matches = list(re.finditer(measurement.pattern, page.text, re.IGNORECASE))
        if len(matches) != 1:
            raise ValueError(f"reviewed measurement phrase missing or ambiguous: {measurement.metric}/{measurement.geography}")
        matched_measurements.append((measurement, matches[0]))
    matched_passages = []
    for passage in source.qualitative_patterns:
        if isinstance(passage, tuple):
            passage = Passage(*passage, "Central London", "Central London")
        matches = list(re.finditer(passage.pattern, page.text))
        if len(matches) != 1:
            raise ValueError(f"reviewed qualitative passage missing or ambiguous: {passage.category}")
        matched_passages.append((passage, matches[0]))
    snapshot = [source.id, source.version, source.published_at, [match.group(0) for _, match in matched_measurements], [match.group(0) for _, match in matched_passages]]
    digest = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, default=str, separators=(",", ":")).encode()).hexdigest()
    doc = Document.model_validate({**doc.model_dump(exclude={"id"}), "published_at": source.published_at, "content_hash": digest, "raw_content_hash": doc.raw_content_hash or doc.content_hash})
    evidence: list[Evidence] = []
    observations: list[Observation] = []
    for measurement, match in matched_measurements:
        value = float(match.group("value").replace(",", ""))
        maximum = 100 if measurement.unit == "%" else 1000 if measurement.metric.endswith("rent") else 100_000_000 if measurement.unit == "sq ft" else 100
        if not 0 <= value <= maximum:
            raise ValueError("reviewed measurement has an out-of-range value")
        qualifications = evidence_qualifications(source, measurement.metric, match.group(0), measurement.period or source.period)
        reason = f"{source.publisher} reviewed HTML contract {source.id} v{source.version}; exact observed phrase, source identity, publication date, reference period, definition and unit checked. Measurement definition. {measurement.definition}." + qualifications
        if source.required_phrases:
            reason += " Source qualifications. " + " ".join(source.caveats)
        if source.accepted_metadata_dates and not source.required_phrases:
            reason += " Source date qualification. " + " ".join(source.caveats)
        native = measurement.native_geography or measurement.geography
        item = Evidence(document_id=doc.id, mode="live", excerpt=match.group(0), published_at=doc.published_at, retrieved_at=doc.retrieved_at, source_url=doc.url, native_geography=native, mapped_geography=measurement.geography, category=measurement.category, validation_status="validated", validation_reason=reason)
        evidence.append(item)
        period = measurement.period or source.period
        if period is None:
            raise ValueError("reviewed measurement requires an exact reference period")
        start, end = period
        observations.append(Observation(mode="live", metric=measurement.metric, value=value, unit=measurement.unit, publisher=source.publisher, definition=measurement.definition, category=measurement.category, native_geography=native, mapped_geography=measurement.geography, period_start=start, period_end=end, evidence_ids=[item.id], dimension=measurement.dimension, segment=measurement.segment, partition=measurement.partition))
    for passage, match in matched_passages:
        qualifications = evidence_qualifications(source, passage.category, match.group(0))
        if source.required_phrases:
            qualifications += " Source qualifications. " + " ".join(source.caveats)
        if passage.qualification:
            qualifications += " Source qualifications. " + passage.qualification
        evidence.append(Evidence(document_id=doc.id, mode="live", excerpt=match.group(0), published_at=doc.published_at, retrieved_at=doc.retrieved_at, source_url=doc.url, native_geography=passage.native_geography, mapped_geography=passage.geography, category=passage.category, statement_kind=passage.kind, validation_status="validated", validation_reason=f"Exact passage from reviewed {source.publisher} {source.source_kind} source {source.id} v{source.version}; qualitative evidence, not a normalized series." + qualifications))
    return ParsedSource(doc, evidence, observations, source.caveats)


_SYNTHETIC_ANALYSIS_SOURCES = (
    MonitoredSource("synthetic-delivery", "https://fixtures.example.invalid/analysis-delivery", "Synthetic delivery", date(2026, 8, 6), Q2, (), (
        Passage("pipeline", r"SYNTHETIC: completions are forecast at 12 million sq ft by end-2028\. 25% of future supply has not yet started construction and remains vulnerable to delay or withdrawal\.", "Central London", "Central London", kind="forecast", qualifications=(ForecastRiskSpec(r"completions are forecast at (?P<quantity>12) million sq ft by end-(?P<year>2028)\.", r"(?P<share>25)% of future supply has not yet started construction and remains vulnerable to delay or withdrawal\.", "million sq ft"),)),
    ), (), version="1", publisher="Synthetic Market Publisher"),
    MonitoredSource("synthetic-prelet", "https://fixtures.example.invalid/analysis-prelet", "Synthetic pre-let", date(2026, 5, 1), Q1, (), (
        Passage("take-up", r"SYNTHETIC: Fictional Meridian's pre-let at Fictional Lantern House completed in Q1 2026 for a 10-year term\.", "Central London", "Central London", facets=("prelets", "leasing"), prelet=PreletSpec(r"SYNTHETIC: (?P<occupier>Fictional Meridian)'s pre-let at (?P<building>Fictional Lantern House) completed in Q1 2026 for a (?P<term>10)-year term\.", Q1)),
    ), (), version="1", publisher="Synthetic Market Publisher"),
)


def fixture_analysis_sources() -> list[tuple[Document, list[Evidence], list[Observation]]]:
    result = []
    for source in _SYNTHETIC_ANALYSIS_SOURCES:
        # These literal synthetic contracts never use live publication identities.
        excerpt = source.qualitative_patterns[0].pattern.replace(r"\.", ".")
        doc = Document(mode="fixture", synthetic=True, url=source.url, title=source.header, publisher=source.publisher, published_at=source.published_at, content_hash=hashlib.sha256(excerpt.encode()).hexdigest())
        leaf = source.qualitative_patterns[0]
        evidence = Evidence(mode="fixture", document_id=doc.id, excerpt=excerpt, source_url=source.url, published_at=doc.published_at, retrieved_at=doc.retrieved_at, native_geography=leaf.native_geography, mapped_geography=leaf.geography, category=leaf.category, statement_kind=leaf.kind, validation_status="validated", validation_reason="Explicit synthetic analysis contract; no live market authority")
        result.append((doc, [evidence], []))
    return result


def reviewed_nonfact_passages(item: Evidence) -> list[tuple[MonitoredSource, Passage, str]]:
    """Finite registered nonfact authority, without promoting retained candidate rows."""
    if item.validation_status == "rejected":
        return []
    matches = []
    for source in (_SYNTHETIC_ANALYSIS_SOURCES if item.mode == "fixture" else MONITORED_SOURCES.values()):
        if str(item.source_url).rstrip("/") != source.url.rstrip("/") or item.published_at != source.published_at:
            continue
        for passage in source.qualitative_patterns:
            if not isinstance(passage, Passage) or not (passage.kind in ("forecast", "interpretation") or source.source_kind == "primary_research_commentary"):
                continue
            if (item.category, item.native_geography, item.mapped_geography) != (passage.category, passage.native_geography, passage.geography):
                continue
            matches.extend((source, passage, match.group(0)) for match in re.finditer(passage.pattern, item.excerpt))
    return matches


def reviewed_source_leaves(item: Evidence):
    """Exact source authority, partitioned between live and synthetic contracts."""
    if item.validation_status != "validated":
        return []
    matches = []
    for source in (_SYNTHETIC_ANALYSIS_SOURCES if item.mode == "fixture" else MONITORED_SOURCES.values()):
        if str(item.source_url).rstrip("/") != source.url.rstrip("/") or item.published_at != source.published_at:
            continue
        for leaf in (*source.measurements, *source.qualitative_patterns):
            if isinstance(leaf, tuple):
                continue
            native = leaf.native_geography or leaf.geography
            if (item.category, item.native_geography, item.mapped_geography) != (leaf.category, native, leaf.geography) or not re.fullmatch(leaf.pattern, item.excerpt, re.I if isinstance(leaf, Measurement) else 0):
                continue
            matches.append((source, leaf))
    return matches


def reviewed_prelet_commitment(item: Evidence) -> PreletCommitment | None:
    resolved = []
    for source, leaf in reviewed_source_leaves(item):
        if not isinstance(leaf, Passage) or leaf.prelet is None:
            continue
        match = re.fullmatch(leaf.prelet.pattern, item.excerpt)
        if match and source.period == leaf.prelet.period:
            resolved.append(PreletCommitment(evidence_id=item.id, source_contract_id=source.id, source_contract_version=source.version, publisher=source.publisher, support_span=item.excerpt, native_geography=leaf.native_geography, occupier=match['occupier'], building=match['building'], term_years=int(match['term']), period_start=leaf.prelet.period[0], period_end=leaf.prelet.period[1]))
    return resolved[0] if len(resolved) == 1 else None


def reviewed_source_qualifications(item: Evidence) -> list[SourceQualification]:
    """Resolve current reviewed contracts against exact validated retained leaves."""
    resolved = []
    for source, leaf in reviewed_source_leaves(item):
        native = leaf.native_geography or leaf.geography
        identity = dict(evidence_id=item.id, source_contract_id=source.id, source_contract_version=source.version)
        for spec in leaf.qualifications:
            if isinstance(spec, ReportedChangeSpec):
                matches = list(re.finditer(spec.pattern, item.excerpt))
                if len(matches) != 1:
                    continue
                resolved.append(SourceReportedChange(**identity, support_span=matches[0].group() if spec.basis == "relative_percent" else item.excerpt, metric=leaf.metric, native_geography=native, baseline_period_start=spec.baseline[0], baseline_period_end=spec.baseline[1], endpoint_period_start=spec.endpoint[0], endpoint_period_end=spec.endpoint[1], change=Quantity(value=float(matches[0]['change']) * spec.sign, unit=spec.unit), basis=spec.basis, requires_comparison=spec.requires_comparison))
            else:
                forecasts, risks = list(re.finditer(spec.forecast_pattern, item.excerpt)), list(re.finditer(spec.risk_pattern, item.excerpt))
                if len(forecasts) != 1 or len(risks) != 1:
                    continue
                forecast, risk = forecasts[0], risks[0]
                resolved.append(ForecastDeliveryRisk(**identity, forecast_span=forecast.group(), risk_span=risk.group(), forecast_quantity=Quantity(value=float(forecast['quantity']), unit=spec.unit), scheme_count=int(forecast['schemes']) if forecast.groupdict().get('schemes') else None, horizon_end=date(int(forecast['year']), 12, 31), unstarted_share=Quantity(value=float(risk['share']), unit="%")))
    return resolved


def comparison_discrepancies(a: Observation, b: Observation, evidence: list[Evidence], calculated_change: float) -> list[ComparisonDiscrepancy]:
    if compatibility(a, b) or abs(calculated_change - (b.value - a.value)) > 1e-6:
        return []
    discrepancies = []
    for item in evidence:
        if item.id not in b.evidence_ids:
            continue
        leaves = reviewed_source_leaves(item)
        if len(leaves) != 1 or not isinstance(leaves[0][1], Measurement):
            continue
        source, measurement = leaves[0]
        for qualification in reviewed_source_qualifications(item):
            if not isinstance(qualification, SourceReportedChange):
                continue
            if (a.metric, b.metric, b.native_geography, a.unit, b.unit, a.publisher, b.publisher) != (qualification.metric, qualification.metric, qualification.native_geography, measurement.unit, measurement.unit, source.publisher, source.publisher):
                continue
            if (a.period_start, a.period_end, b.period_start, b.period_end) != (qualification.baseline_period_start, qualification.baseline_period_end, qualification.endpoint_period_start, qualification.endpoint_period_end):
                continue
            if qualification.basis == "relative_percent":
                if a.value <= 0 or qualification.change.unit != "%":
                    continue
                value, unit = round((b.value - a.value) / a.value * 100, 8), "%"
            else:
                if qualification.change.unit != b.unit:
                    continue
                value, unit = round(calculated_change, 8), b.unit
            if abs(value - qualification.change.value) <= max(1e-8, abs(qualification.change.value) * 1e-6):
                continue
            discrepancies.append(ComparisonDiscrepancy(reported=qualification, calculated_change=Quantity(value=value, unit=unit), from_observation_id=a.id, to_observation_id=b.id, evidence_ids=sorted(set(a.evidence_ids + b.evidence_ids))))
    return discrepancies
