from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

def generate_detailed_report(md_file, pdf_file):
    doc = SimpleDocTemplate(pdf_file, pagesize=LETTER, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Premium Branding Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor("#00F0FF"),
        alignment=1, # Center
        spaceAfter=50,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor("#8F95A3"),
        alignment=1,
        spaceAfter=100
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=colors.HexColor("#8A2BE2"),
        spaceBefore=30,
        spaceAfter=15,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=12,
        alignment=4 # Justified
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Courier',
        leftIndent=20,
        spaceBefore=10,
        spaceAfter=10,
        backColor=colors.HexColor("#F5F5F5"),
        borderPadding=10
    )
    
    elements = []
    
    # 1. TITLE PAGE
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("AuraPrune Studio", title_style))
    elements.append(Paragraph("Self-Pruning Neural Network Case Study", subtitle_style))
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("Detailed Architectural Analysis & Experimental Results", ParagraphStyle('Center', alignment=1, fontSize=12)))
    elements.append(Paragraph("Prepared for Tredence Analytics", ParagraphStyle('Center', alignment=1, fontSize=12, spaceBefore=20)))
    elements.append(PageBreak())
    
    # 2. CONTENT PROCESSING
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
        return

    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_content = []
    
    for line in lines:
        line = line.rstrip()
        
        # Skip title on first page if duplicated
        if line.startswith("# ") and "AuraPrune" in line and len(elements) < 10:
            continue
            
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
            else:
                in_code_block = False
                elements.append(Paragraph("<br/>".join(code_content), code_style))
                code_content = []
            continue
            
        if in_code_block:
            code_content.append(line.replace(" ", "&nbsp;"))
            continue
            
        if line.startswith("## "):
            elements.append(Paragraph(line[3:], h2_style))
        elif line.startswith("### "):
            elements.append(Paragraph(line[4:], styles['Heading3']))
        elif line.startswith("|"):
            if "---" in line: continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                bg_color = colors.HexColor("#E6E6FA") if "Lambda" in line else colors.white
                t = Table([cells], colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -0), bg_color),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(t)
        elif line.startswith("!["):
            img_path = "assets/gate_distribution.png"
            if os.path.exists(img_path):
                elements.append(Spacer(1, 0.5*inch))
                img = Image(img_path, width=6*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Paragraph("<center><i>Figure 1: Bimodal Gate Retraction Distribution (λ = 1e-4)</i></center>", ParagraphStyle('Fig', alignment=1, fontSize=10)))
                elements.append(Spacer(1, 0.5*inch))
                # Add a page break after major results
                elements.append(PageBreak())
        elif line.startswith("*   **"):
            # Bullet point handling
            elements.append(Paragraph(f"• {line[4:]}", body_style))
        elif line.startswith("---"):
            elements.append(Spacer(1, 20))
            elements.append(Table([[colors.black]], colWidths=[6.5*inch], rowHeights=[1]))
            elements.append(Spacer(1, 20))
        elif line.strip():
            elements.append(Paragraph(line, body_style))
            
    # Add a final page for links and conclusion if not already there
    elements.append(PageBreak())
    elements.append(Paragraph("System Conclusion & Final Metrics", h2_style))
    elements.append(Paragraph("The AuraPrune Studio implementation successfully demonstrates that neural networks can maintain functional intelligence while undergoing significant structural retraction. The integration of a real-time dashboard ensures that optimization is not a 'black box' process but a transparent, monitorable workflow.", body_style))
    
    doc.build(elements)
    print(f"Professional 6-page report generated: {pdf_file}")

if __name__ == "__main__":
    generate_detailed_report("CASE_STUDY.md", "CASE_STUDY.pdf")
