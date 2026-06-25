# Agent Skill Declarations

This document declares the skills available to the AI prior authorization agent. Non-programmers can configure or register new skills by writing their interfaces below.

---

## Skill: VerifyCoverageSkill
- Description: Verifies member plan coverage eligibility from enrollment database.
- Inputs: memberId, cptCode
- Outputs: coverageEligible, planType

## Skill: RetrieveGuidelinesSkill
- Description: Queries clinical guidelines catalog for procedure rules.
- Inputs: cptCode
- Outputs: policyId, policyCriteria

## Skill: ExtractClinicalDataSkill
- Description: Parses clinical notes to extract medical triggers and therapy history.
- Inputs: clinicalNotes, cptCode
- Outputs: symptomsDuration, conservativeTherapyWeeks, objectiveFindings

## Skill: EvaluateClinicalCriteriaSkill
- Description: Maps extracted clinical evidence against OPA policy guidelines.
- Inputs: guidelines, evidence, icdCodeMatched
- Outputs: criteriaResultsChecklist

## Skill: GenerateDecisionNoticeSkill
- Description: Drafts the formal correspondence letter with citations and plain language translation.
- Inputs: decision, criteriaResults
- Outputs: noticeLetterDraft

## Skill: CheckAgePatientSkill
- Description: Check the patient sex and make sure it male and in the age group 60 and above
- Inputs: patientDob
- Outputs: checkAgePatientValid
