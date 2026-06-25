import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def build_pdf():
    pdf_filename = "medical_necessity_rules.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            rightMargin=54, leftMargin=54,
                            topMargin=54, bottomMargin=54)
    story = []
    
    # Theme Colors
    primary_color = colors.HexColor("#1e293b") # Slate 800
    secondary_color = colors.HexColor("#4f46e5") # Indigo 600
    text_color = colors.HexColor("#334155") # Slate 700
    
    # Typography Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=primary_color,
        spaceAfter=15,
        leading=26
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        textColor=secondary_color,
        spaceAfter=25,
        leading=14
    )
    
    h1_style = ParagraphStyle(
        'HeadingLevel1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8,
        leading=16
    )
    
    body_style = ParagraphStyle(
        'MainBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=text_color,
        spaceAfter=8,
        leading=13
    )

    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Story flow elements
    story.append(Paragraph("PRIOR AUTHORIZATION CLINICAL POLICY", title_style))
    story.append(Paragraph("Official Medical Necessity Guidelines Catalog & Payer Criteria Rules", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Intro
    intro_text = (
        "This document details the official prior authorization clinical necessity guidelines "
        "defined by Payer Enrollment Services. These guidelines are compiled and verified by "
        "medical directors to govern claims approvals for diagnostic and pharmaceutical requests. "
        "All review workflows must align with the active policy definitions outlined below."
    )
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 12))
    
    # Policy Table
    story.append(Paragraph("Active Policy Code Matrix", h1_style))
    
    data = [
        [Paragraph("Policy ID", bold_body_style), Paragraph("CPT Code", bold_body_style), Paragraph("Service Name", bold_body_style), Paragraph("Eligible Diagnoses (ICD-10)", bold_body_style)],
        [Paragraph("POL-RAD-402", body_style), Paragraph("73721", body_style), Paragraph("MRI Knee Joint", body_style), Paragraph("M25.561, M25.562, M25.569, S83.206A", body_style)],
        [Paragraph("POL-PHARM-809", body_style), Paragraph("J0135", body_style), Paragraph("Adalimumab (Humira)", body_style), Paragraph("M05.79, M06.9", body_style)],
        [Paragraph("POL-EVI-402", body_style), Paragraph("73721", body_style), Paragraph("eviCore MRI Knee", body_style), Paragraph("M25.561, M25.562, M25.569, S83.206A", body_style)]
    ]
    
    t = Table(data, colWidths=[90, 80, 150, 184])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Section: POL-RAD-402
    story.append(Paragraph("Section 1: MRI Knee Joint Clinical Necessity (POL-RAD-402)", h1_style))
    mri_criteria = (
        "Requests for magnetic resonance imaging of the knee joint must satisfy all criteria below: <br/>"
        "1. <b>Symptom Duration:</b> Documented persistent knee pain of at least 6 weeks.<br/>"
        "2. <b>Conservative Care:</b> Documented failure of conservative management (physical therapy, "
        "NSAID regimen, or rest) of at least 6 weeks.<br/>"
        "3. <b>Objective Findings:</b> Physical examination records must report joint tenderness, "
        "swelling, locking, or joint instability."
    )
    story.append(Paragraph(mri_criteria, body_style))
    story.append(Spacer(1, 10))
    
    # Section: POL-PHARM-809
    story.append(Paragraph("Section 2: Biologic Rheumatoid Arthritis Therapy (POL-PHARM-809)", h1_style))
    pharm_criteria = (
        "Requests for specialty biologic agents (e.g. Humira, CPT J0135) must satisfy the following clinical rules: <br/>"
        "1. <b>Diagnosis:</b> Confirmed active Rheumatoid Arthritis (ICD-10 codes M05.79, M06.9).<br/>"
        "2. <b>DMARD Failure:</b> Documented lack of clinical response to conventional Disease-Modifying "
        "Antirheumatic Drugs (e.g., methotrexate) for at least 3 months (12 weeks).<br/>"
        "3. <b>Consultation:</b> Treatment must be prescribed by or in consultation with a rheumatologist."
    )
    story.append(Paragraph(pharm_criteria, body_style))
    story.append(Spacer(1, 10))

    # Section: POL-EVI-402
    story.append(Paragraph("Section 3: UnitedHealthcare / eviCore Advanced Knee Imaging (POL-EVI-402)", h1_style))
    evicore_criteria = (
        "Requests governed by UnitedHealthcare / eviCore musculoskeletal guidelines require prior clinical screening: <br/>"
        "1. <b>Prior Radiographs:</b> Plain radiographs (X-rays) of the knee joint must be performed and documented "
        "after the onset of the current knee symptoms, prior to scheduling advanced imaging (MRI).<br/>"
        "2. <b>Symptom Duration:</b> Persistent pain of at least 6 weeks.<br/>"
        "3. <b>Conservative Care:</b> Documented failure of conservative management (physical therapy or NSAIDs) for at least 6 weeks.<br/>"
        "4. <b>Objective Findings:</b> Documented physical exam findings (tenderness, locking, joint swelling, or instability)."
    )
    story.append(Paragraph(evicore_criteria, body_style))
    story.append(Spacer(1, 20))
    
    # Document footer info
    story.append(Paragraph("Notice: Administrative guidelines mandate that requests not meeting the automated OPA criteria must be escalated to a Human Medical Director for review. Rejections may not be processed by automated agents.", body_style))
    
    doc.build(story)
    print("PDF Rules document compiled successfully.")

if __name__ == "__main__":
    build_pdf()
