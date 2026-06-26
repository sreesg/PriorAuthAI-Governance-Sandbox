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
    input.cptCode == "UNKNOWN"
}

icd_valid {
    input.cptCode == "UNKNOWN"
    input.icd10Code in {}
}

conservative_therapy_met {
    input.cptCode == "UNKNOWN"
    input.extractedEvidence.therapyWeeks >= 6
}

symptoms_duration_met {
    input.cptCode == "UNKNOWN"
    input.extractedEvidence.symptomsDurationWeeks >= 6
}

objective_findings_met {
    input.cptCode == "UNKNOWN"
    input.extractedEvidence.hasObjectiveFindings
}

specialist_consult_met {
    input.cptCode == "UNKNOWN"
}

radiographs_completed_met {
    input.cptCode == "UNKNOWN"
    input.extractedEvidence.hasRadiographs
}

