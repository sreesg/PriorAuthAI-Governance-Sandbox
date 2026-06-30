#!/usr/bin/env python3
"""
seed_graph_db.py — Populate Neo4j with rich clinical graph data for demo patients.

Creates a comprehensive Causal Ontology Graph with:
  - Member nodes (demographics, plan info)
  - Diagnosis/Event nodes (ICD-10 codes, onset dates, status)
  - Medication/Prescription nodes (drugs, doses, start/end dates, outcomes)
  - SDOH Factor nodes (housing, transportation, food security)
  - Policy Rule nodes (CPT criteria, coverage rules)
  - Evidence Source nodes (linked to S3 documents)
  - Provider nodes (NPI, specialty)
  - Relationships: HAS_CONDITION, IS_PRESCRIBED, GOVERNED_BY, EVIDENCED_BY,
                   TRIGGERED_BY, REFERRED_TO, FAILED_THERAPY

USAGE:
  source ./set-aws-profile.sh AKIA... SECRET... TOKEN... us-west-2
  python seed_graph_db.py --neo4j-uri bolt://localhost:7687

  # Or against the deployed Neo4j:
  kubectl port-forward -n beacon svc/neo4j 7687:7687 &
  python seed_graph_db.py

REQUIREMENTS: neo4j
"""

import argparse
import sys
from datetime import datetime

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Missing: pip install neo4j")
    sys.exit(1)


# =============================================================================
# Patient Graph Data — Rich clinical + admin nodes per demo patient
# =============================================================================

PATIENTS_GRAPH = [
    # ─── Robert Chen (MEM-4401) — Lumbar MRI Approve ───
    {
        "member": {
            "member_id": "MEM-4401", "name": "Robert Chen", "dob": "1972-08-15",
            "sex": "male", "plan_id": "PLAN-PPO-2024", "plan_type": "PPO",
            "employer": "TechCorp Industries", "pcp": "Dr. Sarah Lin",
        },
        "diagnoses": [
            {"event_id": "DX-4401-01", "code": "M54.16", "desc": "Radiculopathy, lumbar, left side",
             "onset": "2026-03-15", "status": "active", "severity": "moderate-severe"},
            {"event_id": "DX-4401-02", "code": "M51.16", "desc": "Intervertebral disc degeneration L4-L5",
             "onset": "2026-04-20", "status": "active", "severity": "moderate"},
            {"event_id": "DX-4401-03", "code": "G89.29", "desc": "Other chronic pain",
             "onset": "2026-03-15", "status": "active", "severity": "moderate"},
        ],
        "prescriptions": [
            {"rx_id": "RX-4401-01", "drug": "Ibuprofen", "dose": "800mg TID",
             "start": "2026-03-20", "end": "2026-05-20", "status": "completed", "outcome": "inadequate_response"},
            {"rx_id": "RX-4401-02", "drug": "Gabapentin", "dose": "300mg TID",
             "start": "2026-04-01", "end": "2026-06-01", "status": "completed", "outcome": "inadequate_response"},
            {"rx_id": "RX-4401-03", "drug": "Cyclobenzaprine", "dose": "10mg QHS",
             "start": "2026-04-15", "end": None, "status": "active", "outcome": "partial_response"},
        ],
        "therapies": [
            {"therapy_id": "PT-4401-01", "type": "Physical Therapy", "protocol": "McKenzie",
             "sessions_completed": 16, "sessions_planned": 16, "duration_weeks": 8,
             "start": "2026-04-01", "end": "2026-05-27", "outcome": "failed",
             "notes": "No significant improvement in radicular symptoms despite full compliance"},
        ],
        "providers": [
            {"npi": "1234567890", "name": "Spine Care Associates", "specialty": "Orthopedic Surgery"},
            {"npi": "1111111111", "name": "Dr. Sarah Lin", "specialty": "Internal Medicine"},
            {"npi": "2222222222", "name": "Peak Performance PT", "specialty": "Physical Therapy"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-RAD-501", "name": "Lumbar MRI Policy",
             "criteria": ["Failed 6+ weeks conservative therapy", "Neurological deficit documented",
                         "Physical therapy completed"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-4401-01", "type": "PT Discharge Summary", "date": "2026-05-27"},
            {"evidence_id": "EV-4401-02", "type": "Neurology Consult", "date": "2026-06-01"},
            {"evidence_id": "EV-4401-03", "type": "EMG/NCV Report", "date": "2026-06-05"},
            {"evidence_id": "EV-4401-04", "type": "X-ray Lumbar Spine", "date": "2026-04-20"},
        ],
    },
    # ─── Sarah Martinez (MEM-4402) — Lumbar MRI Escalate ───
    {
        "member": {
            "member_id": "MEM-4402", "name": "Sarah Martinez", "dob": "1985-03-22",
            "sex": "female", "plan_id": "PLAN-HMO-2024", "plan_type": "HMO",
            "employer": "RetailCo Inc", "pcp": "Dr. James Park",
        },
        "diagnoses": [
            {"event_id": "DX-4402-01", "code": "M54.5", "desc": "Low back pain, unspecified",
             "onset": "2026-05-25", "status": "active", "severity": "mild-moderate"},
        ],
        "prescriptions": [
            {"rx_id": "RX-4402-01", "drug": "Acetaminophen", "dose": "500mg PRN",
             "start": "2026-05-25", "end": None, "status": "active", "outcome": "partial_response"},
        ],
        "therapies": [],  # No PT attempted — gap in care
        "providers": [
            {"npi": "2345678901", "name": "Family Medicine Clinic", "specialty": "Family Medicine"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-RAD-501", "name": "Lumbar MRI Policy",
             "criteria": ["Failed 6+ weeks conservative therapy", "Neurological deficit documented"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-4402-01", "type": "Initial Visit Note", "date": "2026-05-25"},
        ],
    },

    # ─── William Johnson (MEM-7701) — PET/CT Approve ───
    {
        "member": {
            "member_id": "MEM-7701", "name": "William Johnson", "dob": "1958-11-03",
            "sex": "male", "plan_id": "PLAN-PPO-2024", "plan_type": "PPO",
            "employer": "Retired", "pcp": "Dr. Maria Santos",
        },
        "diagnoses": [
            {"event_id": "DX-7701-01", "code": "C34.91", "desc": "NSCLC, right lung, unspecified",
             "onset": "2026-05-15", "status": "active", "severity": "Stage IIIA"},
            {"event_id": "DX-7701-02", "code": "R91.1", "desc": "Solitary pulmonary nodule",
             "onset": "2026-04-01", "status": "resolved", "severity": "N/A"},
        ],
        "prescriptions": [
            {"rx_id": "RX-7701-01", "drug": "Ondansetron", "dose": "8mg PRN",
             "start": "2026-05-20", "end": None, "status": "active", "outcome": "supportive"},
        ],
        "therapies": [],
        "providers": [
            {"npi": "3456789012", "name": "Oncology Partners LLC", "specialty": "Medical Oncology"},
            {"npi": "3333333333", "name": "Dr. Williams", "specialty": "Pulmonology"},
            {"npi": "4444444444", "name": "University Pathology", "specialty": "Pathology"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-ONC-220", "name": "PET/CT Oncology Policy",
             "criteria": ["Biopsy-confirmed malignancy", "Initial staging required",
                         "CT findings support indication"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-7701-01", "type": "CT Chest Report", "date": "2026-04-15"},
            {"evidence_id": "EV-7701-02", "type": "CT-Guided Biopsy Procedure", "date": "2026-05-15"},
            {"evidence_id": "EV-7701-03", "type": "Pathology Report", "date": "2026-05-18"},
            {"evidence_id": "EV-7701-04", "type": "Molecular Testing", "date": "2026-05-25"},
            {"evidence_id": "EV-7701-05", "type": "Oncology Consult", "date": "2026-05-28"},
        ],
    },
    # ─── Linda Thompson (MEM-7702) — PET/CT Escalate ───
    {
        "member": {
            "member_id": "MEM-7702", "name": "Linda Thompson", "dob": "1965-07-19",
            "sex": "female", "plan_id": "PLAN-HMO-2024", "plan_type": "HMO",
            "employer": "City Government", "pcp": "Dr. Kevin White",
        },
        "diagnoses": [
            {"event_id": "DX-7702-01", "code": "R91.1", "desc": "Solitary pulmonary nodule",
             "onset": "2026-05-01", "status": "active", "severity": "indeterminate"},
        ],
        "prescriptions": [],
        "therapies": [],
        "providers": [
            {"npi": "4567890123", "name": "Pulmonary Care Group", "specialty": "Pulmonology"},
        ],
        "sdoh_factors": [
            {"sdoh_id": "SDOH-7702-01", "category": "tobacco_use", "desc": "Active smoker, 30 pack-years",
             "origin": "explicit", "confidence": 1.0},
        ],
        "policies": [
            {"policy_id": "POL-ONC-220", "name": "PET/CT Oncology Policy",
             "criteria": ["Biopsy-confirmed malignancy", "Initial staging required"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-7702-01", "type": "Chest CT Report", "date": "2026-05-01"},
        ],
    },

    # ─── David Park (MEM-5501) — Knee Arthroscopy Approve ───
    {
        "member": {
            "member_id": "MEM-5501", "name": "David Park", "dob": "1990-04-12",
            "sex": "male", "plan_id": "PLAN-PPO-2024", "plan_type": "PPO",
            "employer": "Self-employed", "pcp": "Dr. Amy Chang",
        },
        "diagnoses": [
            {"event_id": "DX-5501-01", "code": "S83.201A", "desc": "Lateral meniscus tear, right knee",
             "onset": "2026-02-01", "status": "active", "severity": "Grade 3 complex"},
            {"event_id": "DX-5501-02", "code": "M23.31", "desc": "Other meniscus derangement, right knee",
             "onset": "2026-03-15", "status": "active", "severity": "moderate"},
        ],
        "prescriptions": [
            {"rx_id": "RX-5501-01", "drug": "Naproxen", "dose": "500mg BID",
             "start": "2026-02-05", "end": "2026-04-05", "status": "completed", "outcome": "inadequate_response"},
            {"rx_id": "RX-5501-02", "drug": "Acetaminophen", "dose": "1000mg TID",
             "start": "2026-02-05", "end": None, "status": "active", "outcome": "partial_response"},
        ],
        "therapies": [
            {"therapy_id": "PT-5501-01", "type": "Physical Therapy", "protocol": "Quad strengthening + ROM",
             "sessions_completed": 16, "sessions_planned": 16, "duration_weeks": 8,
             "start": "2026-02-15", "end": "2026-04-12", "outcome": "failed",
             "notes": "Mechanical symptoms persist: locking 2-3x daily, giving way on lateral movements"},
        ],
        "providers": [
            {"npi": "5678901234", "name": "Sports Medicine Surgical Center", "specialty": "Sports Medicine"},
            {"npi": "5555555555", "name": "Dr. Anderson", "specialty": "Orthopedic Surgery"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-SURG-340", "name": "Knee Arthroscopy Policy",
             "criteria": ["MRI confirms meniscus tear", "Failed 6+ weeks PT",
                         "Mechanical symptoms documented", "BMI < 40"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-5501-01", "type": "MRI Right Knee", "date": "2026-03-15"},
            {"evidence_id": "EV-5501-02", "type": "PT Discharge Summary", "date": "2026-04-12"},
            {"evidence_id": "EV-5501-03", "type": "Orthopedic Consult", "date": "2026-04-20"},
        ],
    },
    # ─── Margaret Wilson (MEM-5502) — Knee Arthroscopy Escalate ───
    {
        "member": {
            "member_id": "MEM-5502", "name": "Margaret Wilson", "dob": "1955-12-08",
            "sex": "female", "plan_id": "PLAN-MEDICARE-ADV", "plan_type": "Medicare Advantage",
            "employer": "Retired", "pcp": "Dr. Robert Kim",
        },
        "diagnoses": [
            {"event_id": "DX-5502-01", "code": "M17.11", "desc": "Primary osteoarthritis, right knee",
             "onset": "2024-06-01", "status": "active", "severity": "Kellgren-Lawrence Grade 3"},
            {"event_id": "DX-5502-02", "code": "M23.201", "desc": "Derangement medial meniscus, right",
             "onset": "2025-06-01", "status": "active", "severity": "degenerative horizontal tear"},
        ],
        "prescriptions": [
            {"rx_id": "RX-5502-01", "drug": "Meloxicam", "dose": "15mg daily",
             "start": "2025-01-01", "end": None, "status": "active", "outcome": "partial_response"},
            {"rx_id": "RX-5502-02", "drug": "Triamcinolone injection", "dose": "40mg intra-articular",
             "start": "2025-09-15", "end": "2025-09-15", "status": "completed", "outcome": "temporary_relief"},
        ],
        "therapies": [
            {"therapy_id": "PT-5502-01", "type": "Physical Therapy", "protocol": "Strengthening + flexibility",
             "sessions_completed": 12, "sessions_planned": 12, "duration_weeks": 6,
             "start": "2025-10-01", "end": "2025-11-15", "outcome": "moderate_improvement",
             "notes": "Pain improved but stiffness persists. No true mechanical locking."},
        ],
        "providers": [
            {"npi": "6789012345", "name": "Orthopedic Group Practice", "specialty": "Orthopedic Surgery"},
        ],
        "sdoh_factors": [
            {"sdoh_id": "SDOH-5502-01", "category": "caregiver_availability",
             "desc": "Lives alone, limited support for post-surgical recovery",
             "origin": "inferred", "confidence": 0.65},
        ],
        "policies": [
            {"policy_id": "POL-SURG-340", "name": "Knee Arthroscopy Policy",
             "criteria": ["MRI confirms meniscus tear", "Mechanical symptoms documented",
                         "Age <65 preferred for arthroscopic debridement"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-5502-01", "type": "MRI Right Knee", "date": "2025-06-01"},
            {"evidence_id": "EV-5502-02", "type": "X-ray Bilateral Knees", "date": "2025-03-01"},
        ],
    },

    # ─── Emily Nguyen (MEM-8801) — Dupixent Approve ───
    {
        "member": {
            "member_id": "MEM-8801", "name": "Emily Nguyen", "dob": "1995-06-28",
            "sex": "female", "plan_id": "PLAN-PPO-2024", "plan_type": "PPO",
            "employer": "Design Agency LLC", "pcp": "Dr. Lisa Park",
        },
        "diagnoses": [
            {"event_id": "DX-8801-01", "code": "L20.9", "desc": "Atopic dermatitis, unspecified, severe",
             "onset": "1998-01-01", "status": "active", "severity": "severe (EASI 28, IGA 4)"},
            {"event_id": "DX-8801-02", "code": "L30.9", "desc": "Dermatitis, unspecified",
             "onset": "2020-06-01", "status": "active", "severity": "chronic"},
        ],
        "prescriptions": [
            {"rx_id": "RX-8801-01", "drug": "Triamcinolone 0.1% ointment", "dose": "BID",
             "start": "2025-06-01", "end": "2025-07-15", "status": "completed", "outcome": "failed"},
            {"rx_id": "RX-8801-02", "drug": "Tacrolimus 0.1% ointment", "dose": "BID",
             "start": "2025-07-20", "end": "2025-09-01", "status": "completed", "outcome": "failed"},
            {"rx_id": "RX-8801-03", "drug": "Methotrexate", "dose": "15mg weekly",
             "start": "2025-09-15", "end": "2026-01-15", "status": "discontinued",
             "outcome": "discontinued_adverse_event", "reason": "Elevated liver enzymes (ALT 95)"},
        ],
        "therapies": [],
        "providers": [
            {"npi": "7890123456", "name": "Advanced Dermatology Associates", "specialty": "Dermatology"},
            {"npi": "7777777777", "name": "Dr. Patel", "specialty": "Dermatology (Board Certified)"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-RX-880", "name": "Dupixent (Dupilumab) Policy",
             "criteria": ["Failed topical corticosteroid ≥4 weeks", "Failed calcineurin inhibitor ≥4 weeks",
                         "Failed or intolerant to ≥1 systemic immunosuppressant",
                         "EASI ≥16 or IGA ≥3", "Negative TB screen"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-8801-01", "type": "EASI Scoring Documentation", "date": "2026-02-01"},
            {"evidence_id": "EV-8801-02", "type": "LFT Lab Results (elevated)", "date": "2026-01-10"},
            {"evidence_id": "EV-8801-03", "type": "TB QuantiFERON Gold", "date": "2026-03-01"},
            {"evidence_id": "EV-8801-04", "type": "Photography Documentation", "date": "2026-02-01"},
            {"evidence_id": "EV-8801-05", "type": "Biologic Candidacy Assessment", "date": "2026-03-15"},
        ],
    },

    # ─── James Brown (MEM-8802) — Dupixent Escalate ───
    {
        "member": {
            "member_id": "MEM-8802", "name": "James Brown", "dob": "2012-09-15",
            "sex": "male", "plan_id": "PLAN-PPO-2024", "plan_type": "PPO (dependent)",
            "employer": "Parent's plan", "pcp": "Dr. Michelle Torres",
        },
        "diagnoses": [
            {"event_id": "DX-8802-01", "code": "L20.9", "desc": "Atopic dermatitis, moderate",
             "onset": "2020-01-01", "status": "active", "severity": "moderate (EASI 18, BSA 12%)"},
        ],
        "prescriptions": [
            {"rx_id": "RX-8802-01", "drug": "Hydrocortisone 2.5% cream", "dose": "BID",
             "start": "2026-05-01", "end": "2026-05-15", "status": "completed",
             "outcome": "insufficient_trial", "reason": "Only 2 weeks, not adequate duration"},
        ],
        "therapies": [],
        "providers": [
            {"npi": "8901234567", "name": "Pediatric Dermatology Center", "specialty": "Pediatric Dermatology"},
        ],
        "sdoh_factors": [],
        "policies": [
            {"policy_id": "POL-RX-880", "name": "Dupixent (Dupilumab) Policy",
             "criteria": ["Failed topical corticosteroid ≥4 weeks", "Failed calcineurin inhibitor ≥4 weeks",
                         "Failed systemic immunosuppressant", "Age ≥6"]},
        ],
        "evidence_sources": [
            {"evidence_id": "EV-8802-01", "type": "Initial Derm Visit", "date": "2026-05-01"},
        ],
    },
]


# =============================================================================
# Neo4j Seeding Logic
# =============================================================================

def seed_patient(tx, patient_data):
    """Create all nodes and relationships for a single patient."""
    m = patient_data["member"]
    mid = m["member_id"]

    # Member node
    tx.run("""
        MERGE (m:Member {member_id: $mid})
        SET m.name = $name, m.dob = $dob, m.sex = $sex,
            m.plan_id = $plan_id, m.plan_type = $plan_type,
            m.employer = $employer, m.pcp = $pcp
    """, mid=mid, **{k: m[k] for k in ["name","dob","sex","plan_id","plan_type","employer","pcp"]})

    # Diagnoses
    for dx in patient_data["diagnoses"]:
        tx.run("""
            MERGE (e:Event {event_id: $eid})
            SET e.condition_code = $code, e.description = $desc,
                e.onset_date = $onset, e.status = $status, e.severity = $severity,
                e.type = 'diagnosis'
            WITH e
            MATCH (m:Member {member_id: $mid})
            MERGE (m)-[:HAS_CONDITION]->(e)
        """, eid=dx["event_id"], code=dx["code"], desc=dx["desc"],
             onset=dx["onset"], status=dx["status"], severity=dx["severity"], mid=mid)

    # Prescriptions
    for rx in patient_data["prescriptions"]:
        tx.run("""
            MERGE (e:Event {event_id: $eid})
            SET e.drug = $drug, e.dose = $dose, e.start_date = $start,
                e.end_date = $end, e.status = $status, e.outcome = $outcome,
                e.reason = $reason, e.type = 'prescription'
            WITH e
            MATCH (m:Member {member_id: $mid})
            MERGE (m)-[:IS_PRESCRIBED]->(e)
        """, eid=rx["rx_id"], drug=rx["drug"], dose=rx["dose"],
             start=rx["start"], end=rx.get("end"), status=rx["status"],
             outcome=rx["outcome"], reason=rx.get("reason",""), mid=mid)

    # Therapies (as Events with TRIGGERED_BY from diagnosis)
    for therapy in patient_data.get("therapies", []):
        tx.run("""
            MERGE (e:Event {event_id: $eid})
            SET e.type = 'therapy', e.therapy_type = $ttype, e.protocol = $protocol,
                e.sessions_completed = $completed, e.sessions_planned = $planned,
                e.duration_weeks = $weeks, e.start_date = $start, e.end_date = $end,
                e.outcome = $outcome, e.notes = $notes
            WITH e
            MATCH (m:Member {member_id: $mid})
            MERGE (m)-[:HAS_CONDITION]->(e)
        """, eid=therapy["therapy_id"], ttype=therapy["type"], protocol=therapy["protocol"],
             completed=therapy["sessions_completed"], planned=therapy["sessions_planned"],
             weeks=therapy["duration_weeks"], start=therapy["start"], end=therapy["end"],
             outcome=therapy["outcome"], notes=therapy["notes"], mid=mid)

    # Providers
    for prov in patient_data.get("providers", []):
        tx.run("""
            MERGE (p:Provider {npi: $npi})
            SET p.name = $name, p.specialty = $specialty
            WITH p
            MATCH (m:Member {member_id: $mid})
            MERGE (m)-[:REFERRED_TO]->(p)
        """, npi=prov["npi"], name=prov["name"], specialty=prov["specialty"], mid=mid)

    # SDOH Factors
    for sdoh in patient_data.get("sdoh_factors", []):
        tx.run("""
            MERGE (s:SDOH_Factor {sdoh_id: $sid})
            SET s.category = $cat, s.description = $desc,
                s.origin = $origin, s.confidence = $conf
            WITH s
            MATCH (m:Member {member_id: $mid})
            MERGE (m)-[:HAS_SDOH]->(s)
        """, sid=sdoh["sdoh_id"], cat=sdoh["category"], desc=sdoh["desc"],
             origin=sdoh["origin"], conf=sdoh["confidence"], mid=mid)

    # Policy Rules
    for pol in patient_data.get("policies", []):
        tx.run("""
            MERGE (p:PolicyRule {policy_id: $pid})
            SET p.name = $name, p.criteria = $criteria
            WITH p
            MATCH (m:Member {member_id: $mid})-[:HAS_CONDITION]->(e:Event)
            WITH p, e LIMIT 1
            MERGE (e)-[:GOVERNED_BY]->(p)
        """, pid=pol["policy_id"], name=pol["name"],
             criteria=pol["criteria"], mid=mid)

    # Evidence Sources
    for ev in patient_data.get("evidence_sources", []):
        tx.run("""
            MERGE (es:EvidenceSource {evidence_id: $eid})
            SET es.type = $type, es.date = $date, es.member_id = $mid
            WITH es
            MATCH (m:Member {member_id: $mid})-[:HAS_CONDITION]->(e:Event)
            WITH es, e LIMIT 1
            MERGE (e)-[:EVIDENCED_BY]->(es)
        """, eid=ev["evidence_id"], type=ev["type"], date=ev["date"], mid=mid)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed Neo4j with demo patient graph data")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="beacon-graph-2024")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")
    args = parser.parse_args()

    print(f"Connecting to Neo4j at {args.neo4j_uri}...")
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))

    try:
        driver.verify_connectivity()
        print("✓ Neo4j connected")
    except Exception as e:
        print(f"✗ Cannot connect to Neo4j: {e}")
        sys.exit(1)

    with driver.session() as session:
        if args.clear:
            print("  Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("  ✓ Database cleared")

        # Create constraints
        print("  Creating uniqueness constraints...")
        for label, field in [("Member","member_id"), ("Event","event_id"),
                            ("PolicyRule","policy_id"), ("SDOH_Factor","sdoh_id"),
                            ("EvidenceSource","evidence_id"), ("Provider","npi")]:
            try:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{field} IS UNIQUE")
            except Exception:
                pass

        # Seed each patient
        print(f"\n  Seeding {len(PATIENTS_GRAPH)} patients...")
        for patient_data in PATIENTS_GRAPH:
            mid = patient_data["member"]["member_id"]
            name = patient_data["member"]["name"]
            session.execute_write(seed_patient, patient_data)
            n_nodes = (1 + len(patient_data["diagnoses"]) + len(patient_data["prescriptions"])
                      + len(patient_data.get("therapies",[])) + len(patient_data.get("providers",[]))
                      + len(patient_data.get("sdoh_factors",[])) + len(patient_data.get("policies",[]))
                      + len(patient_data.get("evidence_sources",[])))
            print(f"    ✓ {name} ({mid}): {n_nodes} nodes created")

    # Print summary
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        print(f"\n  Graph summary:")
        for record in result:
            print(f"    {record['label']}: {record['cnt']} nodes")
        rel_result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC")
        for record in rel_result:
            print(f"    [{record['type']}]: {record['cnt']} relationships")

    driver.close()
    print(f"\n✓ Graph seeded with {len(PATIENTS_GRAPH)} patients — rich clinical + admin data")


if __name__ == "__main__":
    main()
