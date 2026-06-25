package prior_auth.policy

default approve = false
default escalate = false

# Automated Approval Rule
approve {
    cpt_valid
    icd_valid
    conservative_therapy_met
    symptoms_duration_met
    objective_findings_met
    specialist_consult_met
    radiographs_completed_met
}

escalate {
    not approve
    has_evidence
}

has_evidence {
    input.clinicalNotes != ""
}

cpt_valid {
    input.cptCode == "73721"
}
cpt_valid {
    input.cptCode == "J0135"
}
cpt_valid {
    input.cptCode == "73721"
}

icd_valid {
    input.cptCode == "73721"
    input.icd10Code in {"M25.561", "M25.562", "M25.569", "S83.206A"}
}
icd_valid {
    input.cptCode == "J0135"
    input.icd10Code in {"M05.79", "M06.9"}
}
icd_valid {
    input.cptCode == "73721"
    input.icd10Code in {"M25.561", "M25.562", "M25.569", "S83.206A"}
}

conservative_therapy_met {
    input.cptCode == "73721"
    input.extractedEvidence.therapyWeeks >= 6
}
conservative_therapy_met {
    input.cptCode == "J0135"
    input.extractedEvidence.therapyWeeks >= 12
}
conservative_therapy_met {
    input.cptCode == "73721"
    input.extractedEvidence.therapyWeeks >= 6
}

symptoms_duration_met {
    input.cptCode == "73721"
    input.extractedEvidence.symptomsDurationWeeks >= 6
}
symptoms_duration_met {
    input.cptCode == "J0135"
    input.extractedEvidence.symptomsDurationWeeks >= 0
}
symptoms_duration_met {
    input.cptCode == "73721"
    input.extractedEvidence.symptomsDurationWeeks >= 6
}

objective_findings_met {
    input.cptCode == "73721"
    input.extractedEvidence.hasObjectiveFindings
}
objective_findings_met {
    input.cptCode == "J0135"
    input.extractedEvidence.hasObjectiveFindings
}
objective_findings_met {
    input.cptCode == "73721"
    input.extractedEvidence.hasObjectiveFindings
}

specialist_consult_met {
    input.cptCode == "73721"
}
specialist_consult_met {
    input.cptCode == "J0135"
    input.extractedEvidence.isRheumatologist
}
specialist_consult_met {
    input.cptCode == "73721"
}

radiographs_completed_met {
    input.cptCode == "73721"
}
radiographs_completed_met {
    input.cptCode == "J0135"
}
radiographs_completed_met {
    input.cptCode == "73721"
    input.extractedEvidence.hasRadiographs
}

knee_icd_codes = {"M25.561", "M25.562", "M25.569", "S83.206A"}
ra_icd_codes = {"M05.79", "M06.9"}
knee_icd_codes = {"M25.561", "M25.562", "M25.569", "S83.206A"}
