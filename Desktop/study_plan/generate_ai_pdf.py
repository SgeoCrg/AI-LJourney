from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
from reportlab.lib.units import cm
from ai_ml_plan_data import AI_ML_PLAN_DATA as PLAN_DATA
import re

def generate_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()

    # Custom Styles
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Title'], fontSize=24, spaceAfter=20))
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=16, alignment=1, spaceAfter=20))
    styles.add(ParagraphStyle(name='ModuleHeader', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='WeekHeader', parent=styles['Heading2'], fontSize=15, textColor=colors.teal, spaceBefore=15, spaceAfter=10))
    
    styles.add(ParagraphStyle(name='DayHeader', parent=styles['Heading3'], fontSize=13, textColor=colors.black, spaceBefore=10, spaceAfter=5, backColor=colors.aliceblue, borderPadding=5))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', textColor=colors.darkslategray, spaceBefore=5))
    styles.add(ParagraphStyle(name='ContentText', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=5))
    styles.add(ParagraphStyle(name='TimeCheck', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Oblique', alignment=2, textColor=colors.grey))

    # Title Page
    story.append(Spacer(1, 4*cm))
    story.append(Paragraph(PLAN_DATA['title'], styles['MainTitle']))
    story.append(Paragraph("Duration: 7 Months (8-10h/week)", styles['SubTitle']))
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Based on the 'AI-ML-Roadmap-from-scratch' Curriculum", styles['SubTitle']))
    story.append(PageBreak())

    # Helper to linkify
    def linkify(text):
        if "<a href" in text: return text # Already HTML
        # Simple regex for URLs
        url_pattern = r'(https?://[^\s)]+)'
        return re.sub(url_pattern, r'<a href="\1" color="blue">\1</a>', text)

    # Content
    for module in PLAN_DATA['modules']:
        story.append(Paragraph(module['name'], styles['ModuleHeader']))
        story.append(Paragraph(f"<i>{module['duration']}</i>", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))

        for week in module['weeks']:
            story.append(Paragraph(f"Week {week['week_num']}: {week['theme']}", styles['WeekHeader']))
            
            for day in week['days']:
                day_content = []
                
                # Header: Day X - Topic
                header_text = f"{day['day']}: {day['topic']}"
                day_content.append(Paragraph(header_text, styles['DayHeader']))
                
                # Theory Section
                day_content.append(Paragraph("Theory & Resources:", styles['SectionHeader']))
                theory_text = day['theory'].strip()
                if "\n" in theory_text and "<br/>" not in theory_text:
                     theory_text = theory_text.replace('\n', '<br/>')
                day_content.append(Paragraph(linkify(theory_text), styles['ContentText']))
                
                # Practice Section
                day_content.append(Paragraph("Practice:", styles['SectionHeader']))
                practice_text = day['practice'].strip()
                if "\n" in practice_text and "<br/>" not in practice_text:
                     practice_text = practice_text.replace('\n', '<br/>')
                day_content.append(Paragraph(linkify(practice_text), styles['ContentText']))
                
                # Time & Checkbox
                time_text = f"Estimated Time: {day['time']} | Completed: [   ]"
                day_content.append(Paragraph(time_text, styles['TimeCheck']))
                
                day_content.append(Spacer(1, 0.5*cm))
                day_content.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=0.5*cm))

                story.append(KeepTogether(day_content))
            
            story.append(PageBreak()) 

    doc.build(story)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    generate_pdf("AI_ML_Study_Plan_7_Months.pdf")
