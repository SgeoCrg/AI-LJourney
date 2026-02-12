from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable, Image
from reportlab.lib.units import cm
from ai_ml_plan_data import AI_ML_PLAN_DATA as PLAN_DATA
import re
import os

# --- Helper Functions ---
def linkify(text):
    if not isinstance(text, str): return ""
    # Only linkify if it's NOT already an HTML link tag
    if "<a href" in text: return text
    url_pattern = r'(https?://[^\s)]+)'
    return re.sub(url_pattern, r'<a href="\1" color="blue">\1</a>', text)

def get_common_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Title'], fontSize=24, spaceAfter=20))
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=16, alignment=1, spaceAfter=20))
    styles.add(ParagraphStyle(name='ModuleHeader', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='WeekHeader', parent=styles['Heading2'], fontSize=15, textColor=colors.teal, spaceBefore=15, spaceAfter=10))
    
    # Detailed Day Styles
    styles.add(ParagraphStyle(name='DayHeader', parent=styles['Heading3'], fontSize=14, textColor=colors.black, spaceBefore=10, spaceAfter=5, backColor=colors.aliceblue, borderPadding=5))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Normal'], fontSize=12, fontName='Helvetica-Bold', textColor=colors.darkslategray, spaceBefore=10))
    styles.add(ParagraphStyle(name='ContentText', parent=styles['Normal'], fontSize=11, leading=15, spaceAfter=8))
    styles.add(ParagraphStyle(name='TimeCheck', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Oblique', alignment=2, textColor=colors.grey))

    # Table Styles
    styles.add(ParagraphStyle(name='CellText', parent=styles['Normal'], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='TableHeader', parent=styles['Normal'], fontSize=10, leading=12, textColor=colors.white, fontName='Helvetica-Bold', alignment=1))
    
    return styles

STYLES = get_common_styles()

# --- 1. General Course Syllabus PDF ---
def generate_course_syllabus(filename="00_AI_ML_Course_Syllabus.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    story = []
    
    story.append(Paragraph(PLAN_DATA['title'], STYLES['MainTitle']))
    story.append(Paragraph("Course Syllabus & Roadmap (7 Months)", STYLES['SubTitle']))
    story.append(Spacer(1, 1*cm))
    
    for module in PLAN_DATA['modules']:
        story.append(Paragraph(module['name'], STYLES['ModuleHeader']))
        story.append(Paragraph(f"<b>Duration:</b> {module['duration']}", STYLES['ContentText']))
        
        # List weeks and themes
        week_data = [["Week", "Theme"]]
        for week in module['weeks']:
            week_data.append([f"Week {week['week_num']}", week['theme']])
            
        t = Table(week_data, colWidths=[3*cm, 12*cm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkslategray),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 1*cm))
        
    doc.build(story)
    print(f"Generated: {filename}")

# --- 2. Module Overview PDF (Table View) ---
def generate_module_overview(module, filename):
    doc = SimpleDocTemplate(filename, pagesize=landscape(A4), topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    story = []
    
    story.append(Paragraph(module['name'], STYLES['MainTitle']))
    story.append(Paragraph(f"Overview & Schedule ({module['duration']})", STYLES['SubTitle']))
    story.append(Spacer(1, 1*cm))

    for week in module['weeks']:
        story.append(Paragraph(f"Week {week['week_num']}: {week['theme']}", STYLES['WeekHeader']))
        
        # Table Header
        data = [["Day", "Topic", "Theory (Brief)", "Practice (Brief)", "Time"]]
        
        for d in week['days']:
            # Strip detailed HTML/newlines for the table view to keep it compact
            # Also strip HTML tags for the table view to avoid complexity
            theory_clean = re.sub('<[^<]+?>', '', d['theory']).replace('\n', ' ')
            if len(theory_clean) > 80: theory_clean = theory_clean[:77] + "..."
            
            practice_clean = re.sub('<[^<]+?>', '', d['practice']).replace('\n', ' ')
            if len(practice_clean) > 80: practice_clean = practice_clean[:77] + "..."

            row = [
                Paragraph(d['day'], STYLES['CellText']),
                Paragraph(d['topic'], STYLES['CellText']),
                Paragraph(theory_clean, STYLES['CellText']),
                Paragraph(practice_clean, STYLES['CellText']),
                Paragraph(d['time'], STYLES['CellText'])
            ]
            data.append(row)

        # Table Style
        col_widths = [2*cm, 5*cm, 9*cm, 9*cm, 2*cm]
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.teal),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        
        # Zebra striping
        for i in range(1, len(data)):
             if i % 2 == 0:
                t.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.aliceblue)]))
                
        story.append(t)
        story.append(Spacer(1, 1*cm))
        story.append(KeepTogether([])) # Just a break point hint
        
    doc.build(story)
    print(f"Generated: {filename}")

# --- 3. Daily Detailed PDF ---
def generate_daily_detailed(day_data, filename, week_info, module_info):
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    story = []
    
    # Header
    story.append(Paragraph(day_data['topic'], STYLES['MainTitle']))
    story.append(Paragraph(f"{module_info} | {week_info} | {day_data['day']}", STYLES['SubTitle']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=1*cm))
    
    # Theory
    story.append(Paragraph("Theory & Concepts", STYLES['SectionHeader']))
    
    # Pre-process content to handle potential XML/HTML issues in ReportLab
    # ReportLab supports a limited subset of XML: <b>, <i>, <u>, <a href>, <br/>, <font>
    # Our data now contains these tags. We should be careful not to escape them, but we need to ensure they are valid.
    # The simplest way is to trust the data input but ensure newlines become <br/>
    
    theory_text = day_data['theory'].strip()
    # If text has no HTML tags at all, we might want to linkify it. 
    # But if it HAS tags (like <a href), our linkify function might break it or double encode it.
    # Our new data has manual <a> tags, so we should SKIP auto-linkify for those fields.
    
    if "\n" in theory_text and "<br/>" not in theory_text:
        theory_text = theory_text.replace('\n', '<br/>')
        
    # We pass theory_text directly. It contains <b> and <a href> tags which ReportLab Paragraph handles.
    try:
        story.append(Paragraph(theory_text, STYLES['ContentText']))
    except:
        # Fallback if parsing fails: strip tags
        clean_text = re.sub('<[^<]+?>', '', theory_text)
        story.append(Paragraph(clean_text, STYLES['ContentText']))
    
    story.append(Spacer(1, 0.5*cm))
    
    # Practice
    story.append(Paragraph("Practice Exercises", STYLES['SectionHeader']))
    practice_text = day_data['practice'].strip()
    if "\n" in practice_text and "<br/>" not in practice_text:
        practice_text = practice_text.replace('\n', '<br/>')

    try:
        story.append(Paragraph(practice_text, STYLES['ContentText']))
    except:
        clean_text = re.sub('<[^<]+?>', '', practice_text)
        story.append(Paragraph(clean_text, STYLES['ContentText']))
    
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=0.5*cm))
    
    story.append(Paragraph(f"Estimated Time: {day_data['time']}", STYLES['ContentText']))
    story.append(Paragraph("Completed: [   ]", STYLES['ContentText']))
    
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("<i>Notes:</i>", STYLES['ContentText']))
    story.append(Paragraph("_"*90, STYLES['ContentText']))
    story.append(Paragraph("_"*90, STYLES['ContentText']))
    story.append(Paragraph("_"*90, STYLES['ContentText']))
    
    doc.build(story)
    print(f"Generated: {filename}")

if __name__ == "__main__":
    # 1. Generate Course Syllabus
    generate_course_syllabus("00_AI_ML_Course_Syllabus.pdf")
    
    # 2. Generate Module Overviews (Months)
    for i, module in enumerate(PLAN_DATA['modules']):
        filename = f"0{i+1}_Month_{i+1}_Overview.pdf"
        generate_module_overview(module, filename)
        
    # 3. Generate detailed Daily PDFs for ALL Months
    global_day_counter = 0

    for module in PLAN_DATA['modules']:
         # Extract month number from name "Month X: ..."
         month_match = re.search(r'Month (\d+)', module['name'])
         if month_match:
            for week in module['weeks']:
                for day in week['days']:
                    global_day_counter += 1

                    # Clean up topic for filename
                    clean_topic = re.sub(r'[^a-zA-Z0-9]', '_', day['topic'])
                    clean_topic = re.sub(r'_+', '_', clean_topic).strip('_')
                    
                    filename_day = f"Day_{str(global_day_counter).zfill(3)}_{clean_topic}.pdf"
                    generate_daily_detailed(day, filename_day, f"Week {week['week_num']}: {week['theme']}", module['name'])
