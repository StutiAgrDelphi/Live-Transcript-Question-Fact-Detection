# Knowledge Base Context

Generated: 2026-08-05 12:04 UTC

## ai.general_knowledge

### What this table contains

This table stores chunked knowledge-base content from a mixed set of source documents used by a live-meeting detection agent. The corpus spans healthcare clinical protocols, IT/security policies and procedures, organizational/contact/reference documents, and a small set of operational or governance materials. Each row represents a chunk from one source document, with metadata preserved to support source tracing and matching.

### Documents and sources present

**Clinical and care protocols:**  
A-01-Adult-Cardiac-Arrest-and-Resuscitation-Protoc, A-02-Recognition-and-Response-to-the-Deteriorating, A-03-Sepsis-Recognition-and-Response-Protocol, A-04-Acute-Stroke-Recognition-and-Activation-Proto, A-05-Anaphylaxis-Recognition-and-Management-Protoc, A-06-Major-Haemorrhage-Protocol, A-07-Isolation-Precautions-and-PPE, A-08-Hand-Hygiene-Protocol, A-09-Peri-operative-Safety-Checklist, A-10-Sharps-Safety-and-Needlestick-Injury-Response, SBI_Health_Policy.

**IT access, device, security, and monitoring policies/procedures:**  
Approved_IT_Policy_CIL_and_Subsidiaries, B-01-Access-Management-Policy, B-01-Access-Management-Policy_v2, B-02-Device-Entitlement-and-Mobile-Device-Policy, B-02-device-entitlement-matrix_v2, B-03-Exception-Request-and-Approval-Procedure, B-03-Exception-Request-and-Approval-Procedure_v2, B-04-Joiner-Mover-Leaver-Procedure, B-04-Joiner-Mover-Leaver-Procedure_v2, B-05-Security-Incident-Reporting-Procedure, B-05-Security-Incident-Reporting-Procedure_v2, B-06-Asset-Liability-and-Disciplinary-Schedule, B-06-asset-liability-disciplinary_v2, B-07-Acceptable-Use-and-Information-Handling-Polic, B-08-Remote-Access-Standard, B-08-Remote-Access-Standard_v2, B-09-Logging-and-Monitoring-Policy, B-09-logging-monitoring-policy_v2, IT-Asset-and-Access-Registers, IT-Asset-and-Access-Registers_v2, Information-Security-Awareness-Training, Information-Security-Awareness-Training_v2.

**Organisation, contact, and operational reference documents:**  
C-01-Organisation-Profile-and-Facilities, C-01-organisation-profile_v2, C-02-Department-Directory-and-Key-Contacts, C-02-department-directory_v2, C-03-Escalation-and-Emergency-Contact-Pathways, C-03-escalation-pathways_v2, C-04-Shift-Handover-Procedure, C-04-shift-handover_v2, C-05-On-Call-Rota-Structure, C-05-on-call-rota_v2, C-06-Visiting-Hours-and-Site-Access-Rules, C-06-visiting-and-site-access_v2, C-07-Badge-and-Physical-Access-Request-Process, C-07-badge-access-request_v2.

**Governance, templates, and supporting/reference material:**  
00-KB-DOCUMENT-STANDARD_v2, README-format-and-coverage, README-format-and-coverage_v2, SYSTEM-PROMPT-guidance, Work Item Discussion_v2.

**Annual reports and broader business/reference content:**  
2023_Annual_Report_Meta, 2024_Annual_Report_Apple_v2, 2024_Annual_Report_Apple_v3, 2024_Annual_Report_Netflix, 2024_Annual_Report_Walmart_v2, 2025_Annual_Report_Nvidia, 1_National_Health_Policy_2017_English_.

### Trigger phrases and vocabulary
cardiac arrest, sepsis recognition, acute stroke, anaphylaxis, major haemorrhage, isolation precautions, hand hygiene, peri-operative safety checklist, sharps safety, needlestick injury, access management, device entitlement, mobile device policy, exception request, joiner mover leaver, security incident reporting, acceptable use, remote access, logging and monitoring, IT asset and access registers, information security awareness training, organisation profile, department directory, escalation and emergency contact pathways, shift handover, on-call rota, visiting hours, site access rules, badge and physical access request process, annual report, national health policy

### What to flag as document_lookup
“Can someone pull up the B-05-Security-Incident-Reporting-Procedure_v2 for this issue?”
“I think the remote access rules are in B-08-Remote-Access-Standard_v2.”
“We should check C-03-Escalation-and-Emergency-Contact-Pathways before we proceed.”
“The handover steps are probably in C-04-Shift-Handover-Procedure.”
“I need the A-03-Sepsis-Recognition-and-Response-Protocol for this patient discussion.”
“Let’s look at 2025_Annual_Report_Nvidia for the latest figures.”
