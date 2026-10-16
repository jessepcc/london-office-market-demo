"""Exact recorded sources for completion callers exercising landlord authoring."""
from pathlib import Path

from london_market.domain import Document


def analysis_sources():
    import json
    from london_market.domain import Evidence
    packet = json.loads((Path(__file__).parents[1]/'evidence/acceptance/requirements-coverage/build-20261011-first-source-packet.json').read_text())
    found = []
    for eid,kind,field in (
        ('ev_627bb5daeb0905ed3544fc02','delivery_uncertainty','forecast_evidence_id'),
        ('ev_8c61432f7e708478bde481fe','lease_commitment','transaction_evidence_id'),
    ):
        evidence = Evidence.model_validate(next(e for e in packet['evidence'] if e['id']==eid))
        document = Document.model_validate(next(d for d in packet['documents'] if d['id']==evidence.document_id))
        found.append((document,evidence,{'kind':'interpretation','interpretation_role':'landlord_analysis','analysis':{'kind':kind,field:eid}}))
    return found


def analysis_intents():
    return [intent for _,_,intent in analysis_sources()]


def seed_analyses(store):
    for document,evidence,_ in analysis_sources():
        store.ingest(document,[evidence],[])


def materialize_analysis(draft):
    from london_market.landlord_analysis import author_landlord_analysis
    eid = getattr(draft.analysis,'forecast_evidence_id',None) or draft.analysis.transaction_evidence_id
    evidence = next(e for _,e,_ in analysis_sources() if e.id==eid)
    return author_landlord_analysis(draft,evidence)
