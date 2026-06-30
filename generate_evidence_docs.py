#!/usr/bin/env python3
"""
generate_evidence_docs.py — Generate synthetic clinical evidence PDFs for demo patients.

Generates ~25 supporting documents per demo patient (200 total) and uploads
directly to S3 without writing to local disk. Documents are tied to the actual
patients from preset_cases.json with coherent clinical timelines.

USAGE:
  source ./set-aws-profile.sh AKIA... SECRET... TOKEN... us-east-1
  pip install boto3 reportlab
  python generate_evidence_docs.py --bucket YOUR_BUCKET

REQUIREMENTS: boto3, reportlab
"""

import argparse
import io
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import boto3
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from reportlab.lib import colors
except ImportError as e:
    print(f"Missing dependency: {e}\nInstall: pip install boto3 reportlab")
    sys.exit(1)


# =============================================================================
# Demo Patient Definitions (from preset_cases.json)
# =============================================================================

DEMO_PATIENTS = [
    {
        "member_id": "MEM-4401", "name": "Robert Chen", "dob": "1972-08-15",
        "age": 52, "sex": "male", "cpt": "72148", "icd10": "M54.16",
        "category": "radiology-lumbar-mri", "outcome": "approve",
        "provider": "Spine Care Associates", "npi": "1234567890",
        "condition": "progressive lower back pain with left L5 radiculopathy",
        "timeline_start": "2026-03-15", "timeline_end": "2026-06-15",
        "clinical_story": {
            "pt_sessions": 16, "pt_duration_weeks": 8, "pt_protocol": "McKenzie",
            "failed_meds": ["ibuprofen 800mg TID", "gabapentin 300mg TID"],
            "exam_findings": ["positive SLR at 30° left", "diminished ankle reflex left",
                             "decreased pinprick L5 dermatome left", "antalgic gait"],
            "imaging": "No prior lumbar MRI. X-ray shows mild degenerative changes L4-L5.",
        },
    },
    {
        "member_id": "MEM-4402", "name": "Sarah Martinez", "dob": "1985-03-22",
        "age": 38, "sex": "female", "cpt": "72148", "icd10": "M54.5",
        "category": "radiology-lumbar-mri", "outcome": "escalate",
        "provider": "Family Medicine Clinic", "npi": "2345678901",
        "condition": "mechanical low back pain for 3 weeks after lifting",
        "timeline_start": "2026-05-25", "timeline_end": "2026-06-15",
        "clinical_story": {
            "pt_sessions": 0, "pt_duration_weeks": 0, "pt_protocol": None,
            "failed_meds": [],
            "exam_findings": ["no radiation to legs", "normal neurological exam",
                             "paravertebral tenderness L3-L5", "full ROM with discomfort"],
            "imaging": "No prior imaging obtained.",
        },
    },
    {
        "member_id": "MEM-7701", "name": "William Johnson", "dob": "1958-11-03",
        "age": 67, "sex": "male", "cpt": "78816", "icd10": "C34.91",
        "category": "oncology-pet-ct", "outcome": "approve",
        "provider": "Oncology Partners LLC", "npi": "3456789012",
        "condition": "non-small cell lung cancer (adenocarcinoma) initial staging",
        "timeline_start": "2026-04-01", "timeline_end": "2026-06-01",
        "clinical_story": {
            "biopsy_date": "2026-05-15", "pathology": "adenocarcinoma TTF-1+",
            "ct_finding": "3.2cm RUL mass with mediastinal lymphadenopathy",
            "markers": "EGFR wild-type, ALK negative, PD-L1 80%",
            "staging": "cT2aN2M0 Stage IIIA",
        },
    },
    {
        "member_id": "MEM-7702", "name": "Linda Thompson", "dob": "1965-07-19",
        "age": 60, "sex": "female", "cpt": "78816", "icd10": "C34.90",
        "category": "oncology-pet-ct", "outcome": "escalate",
        "provider": "Pulmonary Care Group", "npi": "4567890123",
        "condition": "8mm pulmonary nodule found on routine chest CT",
        "timeline_start": "2026-05-01", "timeline_end": "2026-06-10",
        "clinical_story": {
            "biopsy_date": None, "pathology": None,
            "ct_finding": "8mm ground-glass nodule RML",
            "markers": None, "staging": None,
        },
    },
    {
        "member_id": "MEM-5501", "name": "David Park", "dob": "1990-04-12",
        "age": 34, "sex": "male", "cpt": "29881", "icd10": "S83.201A",
        "category": "surgery-knee-arthroscopy", "outcome": "approve",
        "provider": "Sports Medicine Surgical Center", "npi": "5678901234",
        "condition": "complex lateral meniscus tear right knee with mechanical symptoms",
        "timeline_start": "2026-02-01", "timeline_end": "2026-06-01",
        "clinical_story": {
            "pt_sessions": 16, "pt_duration_weeks": 8,
            "mri_finding": "complex tear lateral meniscus posterior horn Grade 3",
            "mechanical_sx": "locking 2-3x daily, gives way with lateral movements",
            "exam": ["McMurray positive", "joint line tenderness laterally", "BMI 24.3"],
        },
    },
    {
        "member_id": "MEM-5502", "name": "Margaret Wilson", "dob": "1955-12-08",
        "age": 70, "sex": "female", "cpt": "29881", "icd10": "M17.11",
        "category": "surgery-knee-arthroscopy", "outcome": "escalate",
        "provider": "Orthopedic Group Practice", "npi": "6789012345",
        "condition": "degenerative medial meniscus tear with osteoarthritis right knee",
        "timeline_start": "2024-06-01", "timeline_end": "2026-06-01",
        "clinical_story": {
            "pt_sessions": 12, "pt_duration_weeks": 6,
            "mri_finding": "degenerative horizontal tear medial meniscus, K-L Grade 3 OA",
            "mechanical_sx": "no true locking, reports stiffness and difficulty with stairs",
            "exam": ["no catching or giving way", "crepitus with ROM", "BMI 32"],
        },
    },
    {
        "member_id": "MEM-8801", "name": "Emily Nguyen", "dob": "1995-06-28",
        "age": 30, "sex": "female", "cpt": "J0875", "icd10": "L20.9",
        "category": "specialty-rx-dupixent", "outcome": "approve",
        "provider": "Advanced Dermatology Associates", "npi": "7890123456",
        "condition": "severe atopic dermatitis since childhood",
        "timeline_start": "2025-06-01", "timeline_end": "2026-06-01",
        "clinical_story": {
            "easi": 28, "iga": 4, "bsa": 35,
            "failed_therapies": [
                "triamcinolone 0.1% ointment BID x 6 weeks",
                "tacrolimus 0.1% ointment BID x 6 weeks",
                "methotrexate 15mg weekly x 4 months (d/c elevated LFTs)",
            ],
            "tb_screen": "QuantiFERON Gold negative 2026-03-01",
        },
    },
    {
        "member_id": "MEM-8802", "name": "James Brown", "dob": "2012-09-15",
        "age": 13, "sex": "male", "cpt": "J0875", "icd10": "L20.9",
        "category": "specialty-rx-dupixent", "outcome": "escalate",
        "provider": "Pediatric Dermatology Center", "npi": "8901234567",
        "condition": "moderate atopic dermatitis",
        "timeline_start": "2026-05-01", "timeline_end": "2026-06-10",
        "clinical_story": {
            "easi": 18, "iga": 3, "bsa": 12,
            "failed_therapies": ["hydrocortisone 2.5% cream x 2 weeks"],
            "tb_screen": None,
        },
    },
]


# =============================================================================
# Document Type Templates per Category
# =============================================================================

DOC_TEMPLATES = {
    "radiology-lumbar-mri": [
        "initial_visit_note", "follow_up_note", "pt_intake_eval",
        "pt_progress_note", "pt_progress_note", "pt_discharge_summary",
        "xray_report", "medication_reconciliation", "neurology_referral",
        "neurology_consult", "emg_report", "pain_management_note",
        "progress_note", "lab_cbc", "lab_inflammatory_markers",
        "disability_assessment", "work_restriction_letter",
        "follow_up_note", "telephone_encounter", "prior_auth_clinical_summary",
        "progress_note", "medication_change_note", "pt_progress_note",
        "follow_up_note", "specialist_recommendation",
    ],
    "oncology-pet-ct": [
        "initial_pcp_visit", "chest_ct_report", "pulmonology_referral",
        "pulmonology_consult", "ct_guided_biopsy_procedure",
        "pathology_report", "oncology_referral", "oncology_consult",
        "lab_cbc", "lab_metabolic_panel", "lab_tumor_markers",
        "molecular_testing_report", "staging_summary",
        "multidisciplinary_tumor_board", "surgical_candidacy_eval",
        "pulmonary_function_test", "cardiology_clearance",
        "follow_up_note", "nursing_assessment", "social_work_note",
        "prior_auth_clinical_summary", "progress_note",
        "telephone_encounter", "lab_cbc", "follow_up_note",
    ],
    "surgery-knee-arthroscopy": [
        "initial_visit_note", "xray_report", "mri_order_note",
        "mri_report", "orthopedic_consult", "pt_intake_eval",
        "pt_progress_note", "pt_progress_note", "pt_progress_note",
        "pt_progress_note", "pt_discharge_summary",
        "follow_up_note", "injection_procedure_note",
        "follow_up_post_injection", "medication_reconciliation",
        "surgical_candidacy_eval", "lab_pre_op",
        "pre_op_clearance", "consent_discussion_note",
        "prior_auth_clinical_summary", "progress_note",
        "activity_log", "telephone_encounter", "follow_up_note",
        "specialist_recommendation",
    ],
    "specialty-rx-dupixent": [
        "initial_derm_visit", "easi_scoring_note", "lab_cbc",
        "lab_ige_eosinophils", "topical_trial_note_1",
        "follow_up_topical_1", "topical_trial_note_2",
        "follow_up_topical_2", "systemic_therapy_start",
        "lab_liver_function", "lab_liver_function_2",
        "systemic_therapy_discontinue", "follow_up_post_systemic",
        "easi_rescoring", "photography_documentation",
        "allergy_consult", "tb_screening_result",
        "biologic_candidacy_assessment", "insurance_appeal_letter",
        "prior_auth_clinical_summary", "medication_reconciliation",
        "quality_of_life_assessment", "follow_up_note",
        "telephone_encounter", "progress_note",
    ],
}


# =============================================================================
# PDF Generation
# =============================================================================

def _build_pdf(elements_fn) -> bytes:
    """Generate a PDF in memory and return bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4)
    meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    elements = elements_fn(h1, h2, body, meta)
    doc.build(elements)
    return buffer.getvalue()


def generate_document(patient: dict, doc_type: str, doc_date: datetime) -> bytes:
    """Generate a single clinical document PDF for a patient."""
    def build(h1, h2, body, meta):
        els = []
        els.append(Paragraph(f"{patient['provider']}", h1))
        els.append(Paragraph(
            f"{doc_type.replace('_', ' ').title()} | "
            f"{doc_date.strftime('%B %d, %Y')} | MRN: {patient['member_id']}",
            meta))
        els.append(Spacer(1, 10))
        els.append(Paragraph(f"Patient: {patient['name']} | DOB: {patient['dob']} | "
                            f"Age: {patient['age']} {patient['sex']}", body))
        els.append(Paragraph(f"Diagnosis: {patient['condition']} ({patient['icd10']})", body))
        els.append(Paragraph(f"CPT: {patient['cpt']} | NPI: {patient['npi']}", body))
        els.append(Spacer(1, 12))

        # Generate content based on doc type and patient story
        content = _generate_content(patient, doc_type, doc_date)
        for section_title, section_body in content:
            els.append(Paragraph(section_title, h2))
            els.append(Paragraph(section_body, body))
            els.append(Spacer(1, 6))

        els.append(Spacer(1, 20))
        els.append(Paragraph(
            f"Electronically signed | Doc ID: {uuid.uuid4()} | "
            f"Category: {patient['category']} | Outcome: {patient['outcome']}",
            meta))
        return els
    return _build_pdf(build)


def _generate_content(patient: dict, doc_type: str, doc_date: datetime) -> list:
    """Generate clinical content sections based on document type and patient."""
    story = patient["clinical_story"]
    name = patient["name"]
    condition = patient["condition"]
    sections = []

    if "visit_note" in doc_type or "progress_note" in doc_type or "follow_up" in doc_type:
        sections.append(("History of Present Illness",
            f"{name} is a {patient['age']}-year-old {patient['sex']} presenting with "
            f"{condition}. " + _get_visit_details(patient, doc_type, doc_date)))
        sections.append(("Assessment",
            f"Primary: {patient['icd10']} — {condition}. " +
            _get_assessment(patient, doc_type)))
        sections.append(("Plan", _get_plan(patient, doc_type)))

    elif "pt_" in doc_type:
        sections.append(("Physical Therapy Note",
            _get_pt_content(patient, doc_type, doc_date)))

    elif "lab_" in doc_type:
        sections.append(("Laboratory Results", _get_lab_content(patient, doc_type)))

    elif "mri_report" in doc_type or "xray_report" in doc_type or "ct_" in doc_type:
        sections.append(("Imaging Report",
            f"Indication: {condition}\n\nFindings: " + _get_imaging(patient, doc_type)))

    elif "pathology" in doc_type:
        sections.append(("Pathology Report", _get_pathology(patient)))

    elif "consult" in doc_type or "referral" in doc_type:
        sections.append(("Consultation Note",
            f"Reason for Referral: {condition}\n\n" + _get_consult(patient, doc_type)))

    elif "procedure" in doc_type or "injection" in doc_type or "biopsy" in doc_type:
        sections.append(("Procedure Note", _get_procedure(patient, doc_type)))

    elif "medication" in doc_type:
        sections.append(("Medication Reconciliation", _get_medications(patient)))

    elif "prior_auth" in doc_type:
        sections.append(("Prior Authorization Clinical Summary",
            _get_pa_summary(patient)))

    elif "easi" in doc_type or "scoring" in doc_type:
        sections.append(("Clinical Scoring", _get_scoring(patient)))

    elif "tb_" in doc_type or "screening" in doc_type:
        sections.append(("Screening Results", _get_screening(patient)))

    else:
        sections.append(("Clinical Note",
            f"Encounter for {name} regarding {condition}. "
            f"See chart for full details. Document type: {doc_type}."))

    return sections


def _get_visit_details(p, doc_type, doc_date):
    story = p["clinical_story"]
    cat = p["category"]
    if cat == "radiology-lumbar-mri":
        findings = story.get("exam_findings", [])
        meds = story.get("failed_meds", [])
        return (f"Symptoms duration: approximately {random.randint(4,16)} weeks. "
                f"Physical exam: {', '.join(findings[:3])}. "
                f"Medications tried: {', '.join(meds) if meds else 'none documented'}. "
                f"PT completed: {story.get('pt_sessions',0)} sessions.")
    elif cat == "oncology-pet-ct":
        return (f"CT findings: {story.get('ct_finding','')}. "
                f"Biopsy: {'confirmed ' + story.get('pathology','') if story.get('biopsy_date') else 'not yet performed'}. "
                f"Staging: {story.get('staging','pending')}.")
    elif cat == "surgery-knee-arthroscopy":
        return (f"MRI: {story.get('mri_finding','')}. "
                f"Mechanical symptoms: {story.get('mechanical_sx','none')}. "
                f"PT sessions completed: {story.get('pt_sessions',0)}. "
                f"Exam: {', '.join(story.get('exam',['unremarkable']))}.")
    elif cat == "specialty-rx-dupixent":
        therapies = story.get("failed_therapies", [])
        return (f"EASI: {story.get('easi','N/A')}, IGA: {story.get('iga','N/A')}, BSA: {story.get('bsa','N/A')}%. "
                f"Failed therapies: {'; '.join(therapies) if therapies else 'limited trials'}. "
                f"TB screen: {story.get('tb_screen','not performed')}.")
    return f"See clinical details for {p['condition']}."


def _get_assessment(p, doc_type):
    if p["outcome"] == "approve":
        return "Medical necessity established. Conservative measures exhausted. Recommend proceeding."
    return "Clinical picture evolving. Additional conservative measures may be warranted before advanced intervention."


def _get_plan(p, doc_type):
    if p["outcome"] == "approve":
        return (f"1. Submit prior authorization for CPT {p['cpt']}\n"
                f"2. Continue current management pending approval\n"
                f"3. Follow-up in 2-4 weeks\n"
                f"4. Patient educated on procedure/treatment risks and benefits")
    return (f"1. Trial of conservative management recommended\n"
            f"2. Consider physical therapy referral\n"
            f"3. Follow-up in 4-6 weeks to reassess\n"
            f"4. If no improvement, may reconsider advanced imaging/intervention")


def _get_pt_content(p, doc_type, doc_date):
    story = p["clinical_story"]
    sessions = story.get("pt_sessions", 0)
    protocol = story.get("pt_protocol", "standard strengthening and ROM")
    if sessions == 0:
        return "No physical therapy has been initiated for this patient."
    if "intake" in doc_type:
        return (f"PT Intake Evaluation. Protocol: {protocol}. "
                f"Planned course: {sessions} sessions over {story.get('pt_duration_weeks',6)} weeks. "
                f"Goals: pain reduction, functional improvement, return to activity.")
    elif "discharge" in doc_type:
        return (f"PT Discharge Summary. Completed {sessions} sessions of {protocol}. "
                f"Patient reports {'minimal improvement' if p['outcome']=='approve' else 'moderate improvement'}. "
                f"{'Recommend advanced intervention.' if p['outcome']=='approve' else 'Continue home exercise program.'}")
    else:
        session_num = random.randint(1, sessions)
        return (f"PT Progress Note — Session {session_num}/{sessions}. Protocol: {protocol}. "
                f"Patient compliance: good. Pain level: {random.randint(4,8)}/10. "
                f"Functional status: {'limited improvement' if p['outcome']=='approve' else 'gradually improving'}.")


def _get_lab_content(p, doc_type):
    labs = []
    if "cbc" in doc_type:
        labs = [f"WBC: {random.uniform(4.5,11.0):.1f} (4.5-11.0)",
                f"Hgb: {random.uniform(11.0,16.0):.1f} (12.0-16.0)",
                f"Plt: {random.randint(150,400)} (150-400)",
                f"Neutrophils: {random.uniform(40,70):.0f}%"]
    elif "inflammatory" in doc_type:
        labs = [f"CRP: {random.uniform(0.2,6.0):.1f} (<0.5)",
                f"ESR: {random.randint(5,55)} (<20)"]
    elif "ige" in doc_type or "eosinophil" in doc_type:
        labs = [f"Total IgE: {random.randint(200,2500)} IU/mL (<100)",
                f"Eosinophils: {random.uniform(3,12):.1f}% (1-4%)"]
    elif "liver" in doc_type:
        elevated = p["outcome"] == "approve"  # elevated LFTs led to d/c of methotrexate
        ast = random.randint(55, 120) if elevated else random.randint(15, 40)
        alt = random.randint(60, 130) if elevated else random.randint(10, 45)
        labs = [f"AST: {ast} (10-40)", f"ALT: {alt} (7-56)",
                f"Alk Phos: {random.randint(40,120)} (44-147)"]
    elif "tumor" in doc_type:
        labs = [f"CEA: {random.uniform(1.0,12.0):.1f} (<3.0)",
                f"CA-125: {random.randint(10,80)} (<35)"]
    elif "pre_op" in doc_type:
        labs = [f"PT: {random.uniform(11,14):.1f}s", f"INR: {random.uniform(0.9,1.2):.1f}",
                f"BMP: within normal limits", f"Type and Screen: completed"]
    else:
        labs = [f"Basic metabolic panel: within normal limits"]
    return f"Specimen collected: {datetime.now().strftime('%m/%d/%Y')}\n\n" + "\n".join(labs)


def _get_imaging(p, doc_type):
    story = p["clinical_story"]
    if "mri" in doc_type:
        return story.get("mri_finding", story.get("imaging", "No significant abnormality."))
    elif "xray" in doc_type:
        return story.get("imaging", "Mild degenerative changes. No acute fracture.")
    elif "ct" in doc_type:
        return story.get("ct_finding", "See report details.")
    return "Imaging findings as noted in clinical record."


def _get_pathology(p):
    story = p["clinical_story"]
    if story.get("pathology"):
        return (f"Specimen: CT-guided lung biopsy {story.get('biopsy_date','')}\n"
                f"Diagnosis: {story['pathology']}\n"
                f"Molecular: {story.get('markers','pending')}")
    return "No pathology specimen available for this patient."


def _get_consult(p, doc_type):
    return (f"Patient referred for evaluation of {p['condition']}. "
            f"After review of available records and examination, "
            f"{'recommend proceeding with requested intervention — medical necessity established.' if p['outcome']=='approve' else 'recommend additional conservative measures before advanced intervention.'}")


def _get_procedure(p, doc_type):
    return (f"Procedure performed for evaluation/treatment of {p['condition']}. "
            f"Patient tolerated procedure well. No complications. "
            f"Plan: follow-up in clinic per schedule.")


def _get_medications(p):
    story = p["clinical_story"]
    meds = story.get("failed_meds", story.get("failed_therapies", []))
    if meds:
        return "Current/Prior Medications:\n" + "\n".join(f"• {m}" for m in meds)
    return "Medication history: limited pharmacotherapy documented."


def _get_pa_summary(p):
    return (f"Prior Authorization Summary for {p['name']} ({p['member_id']})\n"
            f"Requested: CPT {p['cpt']} for {p['condition']}\n"
            f"ICD-10: {p['icd10']}\n"
            f"Provider: {p['provider']} (NPI: {p['npi']})\n"
            f"Clinical justification: See attached documentation.")


def _get_scoring(p):
    story = p["clinical_story"]
    return (f"EASI Score: {story.get('easi','N/A')}\n"
            f"IGA Score: {story.get('iga','N/A')}\n"
            f"BSA Involvement: {story.get('bsa','N/A')}%\n"
            f"DLQI: {random.randint(12,25)}")


def _get_screening(p):
    story = p["clinical_story"]
    tb = story.get("tb_screen")
    if tb:
        return f"TB Screening: {tb}\nHepatitis B: Negative\nHIV: Negative"
    return "Screening labs not yet obtained."


# =============================================================================
# S3 Upload & Main
# =============================================================================

def upload_to_s3(s3_client, bucket: str, key: str, pdf_bytes: bytes):
    """Upload PDF bytes directly to S3."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=pdf_bytes,
                         ContentType="application/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate demo patient evidence PDFs to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="clinical-evidence/", help="S3 prefix")
    parser.add_argument("--count-per-patient", type=int, default=25)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    s3 = boto3.client("s3", **({"region_name": args.region} if args.region else {}))

    # Verify bucket access
    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"✓ Bucket '{args.bucket}' accessible")
    except Exception as e:
        print(f"✗ Cannot access bucket '{args.bucket}': {e}")
        print("  Run: source ./set-aws-profile.sh KEY SECRET TOKEN")
        sys.exit(1)

    total = len(DEMO_PATIENTS) * args.count_per_patient
    print(f"\nGenerating {total} evidence PDFs for {len(DEMO_PATIENTS)} demo patients")
    print(f"  → s3://{args.bucket}/{args.prefix}")
    print("─" * 60)

    uploaded = 0
    for patient in DEMO_PATIENTS:
        mid = patient["member_id"]
        cat = patient["category"]
        templates = DOC_TEMPLATES[cat][:args.count_per_patient]

        # Generate timeline dates spread across patient's history
        start = datetime.strptime(patient["timeline_start"], "%Y-%m-%d")
        end = datetime.strptime(patient["timeline_end"], "%Y-%m-%d")
        span_days = max((end - start).days, 1)

        print(f"\n  [{mid}] {patient['name']} ({patient['outcome'].upper()}) — {len(templates)} docs")

        for i, doc_type in enumerate(templates):
            # Spread documents across timeline
            day_offset = int(span_days * i / len(templates))
            doc_date = start + timedelta(days=day_offset)

            pdf_bytes = generate_document(patient, doc_type, doc_date)
            s3_key = (f"{args.prefix}{mid}/{doc_type}_{doc_date.strftime('%Y%m%d')}"
                     f"_{uuid.uuid4().hex[:8]}.pdf")

            upload_to_s3(s3, args.bucket, s3_key, pdf_bytes)
            uploaded += 1

            if uploaded % 10 == 0:
                print(f"    ... {uploaded}/{total} uploaded")

    print(f"\n{'─' * 60}")
    print(f"✓ Complete: {uploaded} PDFs uploaded to s3://{args.bucket}/{args.prefix}")
    print(f"  Patients: {', '.join(p['member_id'] for p in DEMO_PATIENTS)}")


if __name__ == "__main__":
    main()
