"""
Report Generator – produces a pretty PDF session report using ReportLab.
"""

import os
import time
import datetime
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from reportlab.platypus import Image as RLImage


REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _attention_chart(history, width_cm=14, height_cm=5):
    """Return a ReportLab Image of the attention-score time-series."""
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54),
                           facecolor="#1a1a2e")
    try:
        ax.set_facecolor("#16213e")
        x = np.linspace(0, len(history) / 30 / 60, len(history))   # minutes
        y = np.array(history, dtype=float)

        # Colour-fill zones
        ax.axhspan(80, 100, alpha=0.08, color="#00e676")
        ax.axhspan(55, 80,  alpha=0.08, color="#ffeb3b")
        ax.axhspan(30, 55,  alpha=0.08, color="#ff9800")
        ax.axhspan(0,  30,  alpha=0.08, color="#f44336")

        ax.plot(x, y, color="#00bcd4", linewidth=1.5, alpha=0.9)
        ax.fill_between(x, y, alpha=0.15, color="#00bcd4")

        ax.set_xlim(0, max(x[-1], 0.01))
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (min)", color="#aaa", fontsize=8)
        ax.set_ylabel("Score",      color="#aaa", fontsize=8)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_title("Attention Score Over Session", color="#ccc", fontsize=9, pad=6)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return RLImage(buf, width=width_cm * cm, height=height_cm * cm)
    finally:
        plt.close(fig)


def generate(result: dict, student_name: str = "Student") -> str:
    """
    Generate a PDF report from the final session result dict.
    Returns the file path of the saved PDF.
    """
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = student_name.replace(" ", "_")
    path = os.path.join(REPORT_DIR, f"report_{safe}_{ts}.pdf")

    doc  = SimpleDocTemplate(path, pagesize=A4,
                              rightMargin=2*cm, leftMargin=2*cm,
                              topMargin=2*cm,   bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#0d47a1"),
                                  spaceAfter=4)
    sub_style   = ParagraphStyle("Sub",    parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#555"),
                                  spaceAfter=12, alignment=TA_CENTER)
    hdr_style   = ParagraphStyle("Hdr",    parent=styles["Heading2"],
                                  fontSize=12, textColor=colors.HexColor("#0d47a1"),
                                  spaceBefore=14, spaceAfter=4)
    body_style  = styles["Normal"]
    body_style.fontSize = 9

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Student Attention Monitoring", title_style))
    story.append(Paragraph("Session Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#0d47a1"), spaceAfter=10))

    # ── Summary table ─────────────────────────────────────────────────────────
    # Calculate basic time metrics
    session_sec = result.get("session_sec", 0)
    session_min = session_sec / 60
    dist_sec    = result.get("distracted_sec", 0)
    dist_min    = dist_sec / 60
    att_sec     = max(0.0, session_sec - dist_sec)
    att_min     = att_sec / 60
    
    # 1. Focus Intensity (Average engine score over history)
    full_history = result.get("session_full_history", [])
    if full_history:
        intensity_score = sum(full_history) / len(full_history)
    else:
        intensity_score = result.get("attention_score", 0)
        
    # 2. Focus Persistence (Percentage of time attentive)
    persistence_score = (att_sec / max(session_sec, 0.1)) * 100
    
    # ── Composite Overall Score (Weighted) ──────────────────────────────────
    # 50% Intensity (How focused) + 50% Persistence (How consistently focused)
    avg_score = (intensity_score * 0.5) + (persistence_score * 0.5)
    
    # Cap and sanity check
    avg_score = max(0.0, min(100.0, avg_score))
    last_score = result.get("attention_score", 0)
    
    # Dynamic status based on composite score
    if avg_score >= 85:    status = "Outstandingly Attentive"
    elif avg_score >= 70:  status = "Focused & Attentive"
    elif avg_score >= 50:  status = "Moderately Focused"
    elif avg_score >= 30:  status = "Intermittently Distracted"
    else:                  status = "Severely Distracted"

    story.append(Paragraph("Session Summary", hdr_style))

    data = [
        ["Field", "Value"],
        ["Student Name",         student_name],
        ["Session Date",         datetime.datetime.now().strftime("%d %b %Y  %H:%M")],
        ["Session Duration",     f"{session_min:.1f} min"],
        ["Overall Session Score",f"{avg_score:.1f} / 100"],
        ["Final Attention Score",f"{last_score:.1f} / 100"],
        ["Status",               status],
        ["Attentive Time",       f"{att_min:.1f} min  ({att_min/max(session_min,0.01)*100:.0f}%)"],
        ["Distracted Time",      f"{dist_min:.1f} min  ({dist_min/max(session_min,0.01)*100:.0f}%)"],
        ["Total Blinks",         str(result.get("blinks", 0))],
        ["Avg Eye Aspect Ratio", f"{result.get('ear', 0):.3f}"],
        ["Head Yaw (last)",      f"{result.get('yaw', 0):.1f}°"],
        ["Head Pitch (last)",    f"{result.get('pitch', 0):.1f}°"],
    ]

    tbl = Table(data, colWidths=[6*cm, 10*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), colors.HexColor("#0d47a1")),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("ROWBACKGROUNDS",(0, 2), (-1,-1), [colors.white, colors.HexColor("#e8eaf6")]),
        ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Chart ─────────────────────────────────────────────────────────────────
    history = result.get("session_full_history", [])
    if len(history) >= 2:
        story.append(Paragraph("Attention Score Chart (Full Session)", hdr_style))
        try:
            # Downsample if history is too large (> 5000 points) to prevent performance hits
            if len(history) > 5000:
                step = len(history) // 5000
                history_plot = history[::step]
            else:
                history_plot = history
            
            story.append(_attention_chart(history_plot))
        except Exception as e:
            print(f"Chart Error: {e}")
            story.append(Paragraph("<i>(Chart could not be generated for this session)</i>", body_style))
        story.append(Spacer(1, 0.3*cm))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", hdr_style))
    recs = _recommendations(avg_score, dist_min, session_min, result.get("blinks", 0), session_min)
    for r in recs:
        story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#aaa"), spaceAfter=6))
    story.append(Paragraph(
        "Generated by Student Attention Monitoring System",
        ParagraphStyle("footer", parent=styles["Normal"],
                        fontSize=7, textColor=colors.HexColor("#aaa"),
                        alignment=TA_CENTER)))

    doc.build(story)
    return path


def _recommendations(score, dist_min, total_min, blinks, session_min):
    recs = []
    if score < 60:
        recs.append("The student showed significant attention deficits. Consider shorter sessions or interactive breaks.")
    elif score < 80:
        recs.append("Moderate attention observed. Encourage active participation and reduce distractions.")
    else:
        recs.append("Excellent attention maintained throughout the session. Keep up the great work!")

    if dist_min / max(total_min, 0.01) > 0.3:
        recs.append("More than 30 % of session time was distracted. A structured environment and removing phone/background distractions is advised.")

    if blinks > session_min * 25:
        recs.append("High blink rate detected – possible eye strain. Recommend the 20-20-20 rule (every 20 min, look 20 ft away for 20 s).")
    elif blinks < session_min * 8:
        recs.append("Very low blink rate – student may be straining eyes. Ensure adequate screen distance and brightness.")

    if total_min < 10:
        recs.append("Session was short. Longer monitoring sessions produce more reliable attention data.")

    return recs if recs else ["Attention data within normal ranges. No specific actions required."]
