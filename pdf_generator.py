import io
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_executive_pdf(
    df_summary, issues_list, ai_insights_text, fig_list=None
):
    """Generates an executive PDF report for ARGO Engine.

    Returns a BytesIO buffer ready for st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1E1E2E"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0D6EFD"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2B2B2B"),
    )

    story = []

    
    story.append(
        Paragraph("<b>ARGO ENGINE</b> | Automated Data Assessment", title_style)
    )
    story.append(
        Paragraph(
            "Executive Data Quality & Feature Engineering Report", body_style
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#0D6EFD"),
            spaceAfter=15,
        )
    )

    
    story.append(Paragraph("1. Dataset Overview", h2_style))
    overview_data = [
        ["Metric", "Value"],
        ["Total Rows", str(df_summary.get("rows", "N/A"))],
        ["Total Columns", str(df_summary.get("cols", "N/A"))],
        ["Duplicates Found", str(df_summary.get("duplicates", "N/A"))],
        ["Missing Cells", str(df_summary.get("missing_total", "N/A"))],
    ]
    t_overview = Table(overview_data, colWidths=[200, 300])
    t_overview.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#F1F3F5"),
            ),  # هيدر الجدول
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E1E2E")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ])
    )
    story.append(t_overview)
    story.append(Spacer(1, 15))

    
    story.append(Paragraph("2. Detected Quality Issues", h2_style))
    if issues_list:
        issues_table_data = [["Issue Type", "Severity", "Description"]]
        for issue in issues_list:
            issues_table_data.append([
                Paragraph(str(issue.get("type", "Issue")), body_style),
                Paragraph(str(issue.get("severity", "Info")), body_style),
                Paragraph(str(issue.get("desc", "")), body_style),
            ])

        t_issues = Table(issues_table_data, colWidths=[120, 80, 300])
        t_issues.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED4DA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(t_issues)
    else:
        story.append(Paragraph("No critical issues detected.", body_style))

    story.append(Spacer(1, 15))

    
    if fig_list:
        story.append(Paragraph("3. Data Distribution Charts", h2_style))
        for fig in fig_list:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
            img_buf.seek(0)
            story.append(Image(img_buf, width=450, height=220))
            story.append(Spacer(1, 10))

    
    story.append(
        Paragraph("4. AI Copilot Recommendations & Pipeline Insights", h2_style)
    )
    
    formatted_ai_text = ai_insights_text.replace("\n", "<br/>")
    story.append(Paragraph(formatted_ai_text, body_style))

    
    doc.build(story)
    buffer.seek(0)
    return buffer