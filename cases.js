// Mock databases for the Prior Authorization Agent

export const CLINICAL_GUIDELINES = {
  "73721": { // CPT code for MRI Lower Joint (Knee)
    policyId: "POL-RAD-402",
    policyName: "Magnetic Resonance Imaging (MRI) of the Knee Joint",
    effectiveDate: "2026-01-01",
    requiredIcd10: ["M25.561", "M25.562", "M25.569", "S83.206A"], // Knee pain codes, Meniscal tear
    criteria: [
      {
        id: "CRIT-1",
        text: "Patient has persistent knee pain for at least 6 weeks.",
        type: "symptom_duration",
        requiredValue: 6 // weeks
      },
      {
        id: "CRIT-2",
        text: "Patient has tried conservative therapy (e.g., physical therapy, NSAIDs, rest) for at least 6 weeks.",
        type: "conservative_treatment",
        requiredValue: 6 // weeks
      },
      {
        id: "CRIT-3",
        text: "Objective clinical findings are documented (e.g., joint swelling, instability, locking, or joint line tenderness).",
        type: "objective_findings",
        requiredValue: true
      },
      {
        id: "CRIT-4",
        text: "Patient has completed plain radiographs (X-rays) of the knee joint.",
        type: "radiographs_completed",
        requiredValue: true
      }
    ]
  },
  "J0135": { // CPT/HCPCS code for Adalimumab (Humira) - Biologic Drug
    policyId: "POL-PHARM-809",
    policyName: "Biologic Therapy for Rheumatoid Arthritis",
    effectiveDate: "2026-03-01",
    requiredIcd10: ["M05.79", "M06.9"], // Rheumatoid Arthritis
    criteria: [
      {
        id: "CRIT-1",
        text: "Patient has a confirmed diagnosis of severe active Rheumatoid Arthritis (RA).",
        type: "diagnosis_confirmation",
        requiredValue: true
      },
      {
        id: "CRIT-2",
        text: "Patient has tried and failed at least one conventional DMARD (e.g., methotrexate, sulfasalazine) for at least 3 months.",
        type: "dmard_failure",
        requiredValue: true
      },
      {
        id: "CRIT-3",
        text: "Prescribed by or in consultation with a rheumatologist.",
        type: "specialist_consult",
        requiredValue: true
      }
    ]
  }
};

export const MEMBER_BENEFITS = {
  "MEM-9981": {
    name: "John H. Doe", // PHI alert
    ssn: "000-12-3456",  // PHI alert
    dob: "1984-05-12",   // PHI alert
    status: "Active",
    planType: "PPO Platinum",
    copay: 20,
    deductibleMet: true,
    coveredCategories: ["Radiology", "Pharmacy", "Specialty Drugs"]
  },
  "MEM-2210": {
    name: "Jane Smith",  // PHI alert
    ssn: "111-23-4567",  // PHI alert
    dob: "1978-11-23",   // PHI alert
    status: "Active",
    planType: "HMO Bronze",
    copay: 50,
    deductibleMet: false,
    coveredCategories: ["Radiology"] // Note: NO "Pharmacy" or "Specialty Drugs" category!
  }
};

export const PRESET_CASES = [
  {
    id: "case-1",
    title: "Case 1: Knee MRI - Auto-Approved",
    description: "Knee MRI request that meets all medical necessity guidelines.",
    request: {
      memberId: "MEM-9981",
      patientName: "John H. Doe",
      patientSsn: "000-12-3456",
      patientDob: "1984-05-12",
      cptCode: "73721",
      icd10Code: "M25.561", // Right knee pain
      providerNpi: "1992019202",
      providerName: "Ortho Specialists Inc",
      clinicalNotes: "Patient presents with persistent right knee pain for 8 weeks after a minor twisting injury. Has completed 6 weeks of structured physical therapy and tried daily ibuprofen without relief. Joint line tenderness is positive on examination. Plain radiographs (X-rays) of the knee joint were performed and showed minor joint spacing but no acute fractures."
    }
  },
  {
    id: "case-2",
    title: "Case 2: Knee MRI - Escalated to Human Review",
    description: "Knee MRI request where the conservative therapy guideline is NOT met.",
    request: {
      memberId: "MEM-9981",
      patientName: "John H. Doe",
      patientSsn: "000-12-3456",
      patientDob: "1984-05-12",
      cptCode: "73721",
      icd10Code: "M25.562", // Left knee pain
      providerNpi: "1992019202",
      providerName: "Ortho Specialists Inc",
      clinicalNotes: "Patient has left knee pain for 2 weeks after slip and fall. No physical therapy completed yet. Patient is asking for MRI to rule out meniscus tear quickly due to intense pain. No plain radiographs performed."
    }
  },
  {
    id: "case-3",
    title: "Case 3: Humira - Specialty Coverage Error & Code Mismatch",
    description: "Humira drug requested for an HMO member without Pharmacy coverage and mismatched diagnosis code.",
    request: {
      memberId: "MEM-2210",
      patientName: "Jane Smith",
      patientSsn: "111-23-4567",
      patientDob: "1978-11-23",
      cptCode: "J0135", // Humira CPT
      icd10Code: "M25.561", // Mismatched code: Knee pain instead of Rheumatoid Arthritis!
      providerNpi: "1882773641",
      providerName: "RheumCare Group",
      clinicalNotes: "Patient has chronic knee pain, requesting biologic therapy for severe joints inflammation. Consulted by Dr. Evans (Rheumatologist). Has been taking methotrexate for 6 months."
    }
  },
  {
    id: "case-4",
    title: "Case 4: Custom Sandbox Case",
    description: "A customizable template to edit codes, clinical notes, and patient details.",
    request: {
      memberId: "MEM-9981",
      patientName: "John H. Doe",
      patientSsn: "000-12-3456",
      patientDob: "1984-05-12",
      cptCode: "73721",
      icd10Code: "M25.561",
      providerNpi: "1992019202",
      providerName: "Ortho Specialists Inc",
      clinicalNotes: "Replace this clinical text with your own test case. Ensure to include therapy details, durations, and symptoms to watch the AI evaluate them."
    }
  }
];
