# Clinical Necessity Policy Declarations

This document holds the human-readable prior authorization clinical criteria rules. Non-programmer medical directors can update this file to configure guidelines. The SkillJuror OPA compiler parses this markdown to generate active Rego rules.

---

## Policy: POL-RAD-402 (MRI Knee Joint)
- CPT Code: 73721
- Allowed ICD-10 Diagnosis Codes: M25.561, M25.562, M25.569, S83.206A
- Minimum Symptom Duration: 6 weeks
- Minimum Conservative Therapy: 6 weeks
- Objective Findings Required: True
- Specialist Consultation Required: False
- Plain Radiographs Completed: False

## Policy: POL-PHARM-809 (Biologic Therapy for RA)
- CPT Code: J0135
- Allowed ICD-10 Diagnosis Codes: M05.79, M06.9
- Minimum Symptom Duration: 0 weeks
- Minimum Conservative Therapy: 12 weeks
- Objective Findings Required: True
- Specialist Consultation Required: True
- Plain Radiographs Completed: False

## Policy: POL-EVI-402 (eviCore Knee MRI Guidelines)
- CPT Code: 73721
- Allowed ICD-10 Diagnosis Codes: M25.561, M25.562, M25.569, S83.206A
- Minimum Symptom Duration: 6 weeks
- Minimum Conservative Therapy: 6 weeks
- Objective Findings Required: True
- Specialist Consultation Required: False
- Plain Radiographs Completed: True
