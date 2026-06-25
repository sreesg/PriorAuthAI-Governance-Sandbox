export function VerifyNpiStatusSkill(context, agent) {
  const npi = context.request.providerNpi;
  // Simple validation: NPI should be a 10‑digit numeric string
  const isValid = /^[0-9]{10}$/.test(npi);
  if (isValid) {
    context.npiStatus = "Valid";
    agent.logTrace("skill", "VerifyNpiStatusSkill", `NPI ${npi} passed validation.`, "success");
  } else {
    context.npiStatus = "Invalid";
    agent.logTrace("skill", "VerifyNpiStatusSkill", `NPI ${npi} failed validation.`, "fail");
  }
}
