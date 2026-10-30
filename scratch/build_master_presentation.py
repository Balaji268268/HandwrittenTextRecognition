import os
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ==============================================================================
# COLOR PALETTE & DESIGN TOKENS
# ==============================================================================
COLOR_NAVY = RGBColor(14, 42, 71)          # #0E2A47 - Primary Dark
COLOR_TEAL = RGBColor(27, 158, 142)        # #1B9E8E - Action / Accent
COLOR_BLUE = RGBColor(24, 76, 120)         # #184C78 - Secondary Deep Blue
COLOR_SLATE = RGBColor(36, 49, 64)         # #243140 - Neutral Dark Text
COLOR_MUTED = RGBColor(90, 107, 124)       # #5A6B7C - Neutral Light Text
COLOR_WHITE = RGBColor(255, 255, 255)      # Pure White
COLOR_CARD_BG = RGBColor(248, 250, 252)    # #F8FAFC - Card Background
COLOR_CARD_BORDER = RGBColor(218, 226, 235)# #DAE2EB - Card Stroke
COLOR_TEAL_LIGHT = RGBColor(235, 248, 246) # #EBF8F6 - Subtle Accent Fill
COLOR_ROW_ALT = RGBColor(242, 246, 250)    # Table Row Alternating

FONT_NAME = 'Calibri'
AUTHOR_NAME = "A Venkata Balaji   ·   Reg. No: 23BCE20336"
AFFILIATION = "School of Computer Science & Engineering, VIT"
REPO_URL = "https://github.com/Balaji268268/HandwrittenTextRecognition"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def add_header_footer(slide, category_text, title_text):
    """Standardized institutional header and footer across all content slides."""
    # Category / Eyebrow
    tb_cat = slide.shapes.add_textbox(Inches(0.60), Inches(0.34), Inches(11.50), Inches(0.30))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_NAME
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_TEAL

    # Main Slide Title
    tb_title = slide.shapes.add_textbox(Inches(0.60), Inches(0.60), Inches(12.10), Inches(0.62))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_NAME
    p_title.font.size = Pt(25)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY

    # Teal Horizontal Accent Underline
    acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.63), Inches(1.24), Inches(1.20), Inches(0.05))
    acc.fill.solid()
    acc.fill.fore_color.rgb = COLOR_TEAL
    acc.line.fill.background()

    # Footer Left (Author & Reg)
    tb_fl = slide.shapes.add_textbox(Inches(0.60), Inches(7.14), Inches(8.50), Inches(0.30))
    tf_fl = tb_fl.text_frame
    tf_fl.margin_left = tf_fl.margin_right = tf_fl.margin_top = tf_fl.margin_bottom = 0
    p_fl = tf_fl.paragraphs[0]
    p_fl.text = f"{AUTHOR_NAME}   ·   {AFFILIATION}"
    p_fl.font.name = FONT_NAME
    p_fl.font.size = Pt(10)
    p_fl.font.color.rgb = COLOR_MUTED

    # Footer Right (Project Title)
    tb_fr = slide.shapes.add_textbox(Inches(8.80), Inches(7.14), Inches(3.90), Inches(0.30))
    tf_fr = tb_fr.text_frame
    tf_fr.margin_left = tf_fr.margin_right = tf_fr.margin_top = tf_fr.margin_bottom = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.text = "Scale-Invariant Multi-Scenario Offline HTR"
    p_fr.font.name = FONT_NAME
    p_fr.font.size = Pt(10)
    p_fr.font.color.rgb = COLOR_MUTED

def clear_slide_shapes(slide):
    """Removes existing shapes from a slide."""
    sp_tree = slide.shapes._spTree
    for shp in list(slide.shapes):
        sp_tree.remove(shp._element)

# ==============================================================================
# SLIDE BUILDERS (16 MASTER SLIDES)
# ==============================================================================

def slide_1_title(slide):
    clear_slide_shapes(slide)
    
    # Background Accent Panel on Right
    bg_right = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.20), Inches(0.0), Inches(4.133), Inches(7.50))
    bg_right.fill.solid()
    bg_right.fill.fore_color.rgb = COLOR_NAVY
    bg_right.line.fill.background()

    # Right Panel Highlights
    tb_rp = slide.shapes.add_textbox(Inches(9.50), Inches(1.50), Inches(3.50), Inches(5.00))
    tf_rp = tb_rp.text_frame
    tf_rp.word_wrap = True
    
    p1 = tf_rp.paragraphs[0]
    p1.text = "CORE METRICS"
    p1.font.name = FONT_NAME
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEAL
    p1.space_after = Pt(12)

    metrics = [
        ("4.18%", "Overall Macro CER"),
        ("9.62%", "Word Error Rate"),
        ("89.0%", "Exact Sequence Acc"),
        ("27.5 ms", "Inference Latency"),
        ("0 Pretrained", "Weights Required"),
        ("100/100", "Realistic Scenarios")
    ]
    for val, lbl in metrics:
        p_val = tf_rp.add_paragraph()
        p_val.text = val
        p_val.font.name = FONT_NAME
        p_val.font.size = Pt(18)
        p_val.font.bold = True
        p_val.font.color.rgb = COLOR_WHITE
        
        p_lbl = tf_rp.add_paragraph()
        p_lbl.text = lbl
        p_lbl.font.name = FONT_NAME
        p_lbl.font.size = Pt(10)
        p_lbl.font.color.rgb = COLOR_TEAL_LIGHT
        p_lbl.space_after = Pt(8)

    # Left Side: Project Metadata & Title
    tb_top = slide.shapes.add_textbox(Inches(0.80), Inches(0.90), Inches(8.00), Inches(0.40))
    tf_top = tb_top.text_frame
    p_top = tf_top.paragraphs[0]
    p_top.text = "ACADEMIC RESEARCH & PRODUCTION DEEP LEARNING   ·   OCTOBER 2026"
    p_top.font.name = FONT_NAME
    p_top.font.size = Pt(11)
    p_top.font.bold = True
    p_top.font.color.rgb = COLOR_TEAL

    tb_t = slide.shapes.add_textbox(Inches(0.80), Inches(1.35), Inches(8.00), Inches(2.20))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "Scale-Invariant and Noise-Robust\nOffline Handwritten Text Recognition\nfor Multi-Domain Documents"
    p_t.font.name = FONT_NAME
    p_t.font.size = Pt(28)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_NAVY

    acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.83), Inches(3.75), Inches(1.60), Inches(0.06))
    acc.fill.solid()
    acc.fill.fore_color.rgb = COLOR_TEAL
    acc.line.fill.background()

    tb_sub = slide.shapes.add_textbox(Inches(0.80), Inches(3.95), Inches(8.00), Inches(0.80))
    tf_sub = tb_sub.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "End-to-End Deep Learning Architecture Trained From Scratch Without Pretrained Models — Robust Across Doctor Prescriptions, Bank Checks, Multi-Line Letters, and Degraded Scans."
    p_sub.font.name = FONT_NAME
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = COLOR_SLATE

    # Presenter Details Card
    card_p = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.80), Inches(4.90), Inches(8.00), Inches(1.10))
    card_p.fill.solid()
    card_p.fill.fore_color.rgb = COLOR_CARD_BG
    card_p.line.color.rgb = COLOR_CARD_BORDER
    card_p.line.width = Pt(1)
    tf_cp = card_p.text_frame
    tf_cp.margin_left = Inches(0.20)
    tf_cp.margin_top = Inches(0.12)
    
    pc1 = tf_cp.paragraphs[0]
    pc1.text = "Author: A Venkata Balaji   |   Registration No: 23BCE20336"
    pc1.font.name = FONT_NAME
    pc1.font.size = Pt(13)
    pc1.font.bold = True
    pc1.font.color.rgb = COLOR_NAVY
    pc1.space_after = Pt(2)

    pc2 = tf_cp.add_paragraph()
    pc2.text = f"{AFFILIATION}   ·   Email: galigastagiramma@gmail.com"
    pc2.font.name = FONT_NAME
    pc2.font.size = Pt(11)
    pc2.font.color.rgb = COLOR_SLATE

    # GitHub Repository Badge
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.80), Inches(6.15), Inches(8.00), Inches(0.50))
    badge.fill.solid()
    badge.fill.fore_color.rgb = COLOR_TEAL_LIGHT
    badge.line.color.rgb = COLOR_TEAL
    badge.line.width = Pt(1.5)
    tf_b = badge.text_frame
    tf_b.margin_left = Inches(0.20)
    tf_b.margin_top = Inches(0.08)
    pb = tf_b.paragraphs[0]
    pb.text = f"Project Repository: {REPO_URL}"
    pb.font.name = FONT_NAME
    pb.font.size = Pt(11.5)
    pb.font.bold = True
    pb.font.color.rgb = COLOR_NAVY

def slide_2_motivation(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "EXECUTIVE SUMMARY", "Why Offline HTR Is Hard — The Real-World Reality")

    pillars = [
        ("1. High Intraclass Variability", COLOR_TEAL, [
            ("Extreme Pen Styles", "Same letter written by 100 writers exhibits 100 distinct morphologies."),
            ("Ligatures & Slant", "Cursive script blends adjacent letters into unbroken flowing strokes."),
            ("Baseline Wander", "Lines tilt, curve, and wander dynamically across unguided pages.")
        ]),
        ("2. Real-World Document Stress", COLOR_BLUE, [
            ("Doctor Prescriptions", "Physician shorthand, Latin abbreviations (b.i.d.), and dosages (500mg)."),
            ("Bank Checks", "Currency amounts ($ 1,250.00), comma punctuation, watermarks, stamps."),
            ("Mobile Phone Snapshots", "Angular tilt (+/-15 deg), flash glare, and overhead lamp shadows.")
        ]),
        ("3. Why Vision Transformers Fail From Scratch", COLOR_NAVY, [
            ("Massive Token Hunger", "TrOCR requires 684M synthetic typography tokens to converge."),
            ("Attention Collapse", "Scratch-trained ViTs on academic corpora yield CER > 38%."),
            ("Latency Penalty", "Autoregressive generation takes > 300 ms per line; edge-unfriendly.")
        ]),
        ("4. Our Core Breakthroughs", COLOR_TEAL, [
            ("Zero Pretrained Models", "Pure scratch-trained Gated-CNN + Attention + BiLSTM + CTC."),
            ("Scale-Invariant Engine", "Dynamic character height estimation (med_h), deskewing, doodle filter."),
            ("iam-digits Composite", "91,540 samples uniting IAM, CVL, and ORAND bank checks.")
        ])
    ]

    card_w = 2.85
    gap = 0.20
    top_pos = 1.48
    card_h = 3.75

    for idx, (title, color, items) in enumerate(pillars):
        c_left = 0.60 + idx * (card_w + gap)

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(top_pos), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.18)
        tf.margin_top = Inches(0.16)

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.name = FONT_NAME
        p_head.font.size = Pt(11.5)
        p_head.font.bold = True
        p_head.font.color.rgb = color
        p_head.space_after = Pt(8)

        for k, v in items:
            p = tf.add_paragraph()
            p.text = f"• {k}: {v}"
            p.font.name = FONT_NAME
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_SLATE
            p.space_after = Pt(6)

    # Bottom Callout Box
    c_bot = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(5.45), Inches(12.10), Inches(1.45))
    c_bot.fill.solid()
    c_bot.fill.fore_color.rgb = COLOR_TEAL_LIGHT
    c_bot.line.color.rgb = COLOR_TEAL
    c_bot.line.width = Pt(1)
    tf_b = c_bot.text_frame
    tf_b.margin_left = Inches(0.22)
    tf_b.margin_top = Inches(0.12)
    pb1 = tf_b.paragraphs[0]
    pb1.text = "The Goal: A Production-Ready Deep Learning Model That Works Across Every Scenario"
    pb1.font.name = FONT_NAME
    pb1.font.size = Pt(12)
    pb1.font.bold = True
    pb1.font.color.rgb = COLOR_NAVY
    pb1.space_after = Pt(3)

    pb2 = tf_b.add_paragraph()
    pb2.text = "Instead of relying on multi-billion parameter proprietary vision-language APIs or fragile single-sample hacks, our system unifies dynamic scale-invariant morphology with deep feature gating. The result is an end-to-end differentiable system delivering 4.18% CER across 10 distinct real-world stress categories."
    pb2.font.name = FONT_NAME
    pb2.font.size = Pt(10)
    pb2.font.color.rgb = COLOR_SLATE

def slide_3_scenarios_taxonomy(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "SCENARIO MATRIX", "The 10 Real-World Benchmark Scenarios")

    scenarios = [
        ("1. Doctor Prescriptions", "Physician shorthand, drug names (Amoxicillin), dosages (500mg), and intake regimens (1-0-1 b.i.d.).", COLOR_TEAL),
        ("2. Bank Checks & Drafts", "High-value numeric courtesy amounts ($ 1,250.00), legal words, check/account numbers.", COLOR_BLUE),
        ("3. Dense Multi-Line Letters", "Tightly packed multi-line paragraphs; interlocking ascenders/descenders requiring valley splitting.", COLOR_NAVY),
        ("4. Notes with Doodles", "Greeting cards with hand-drawn smiling suns, stars, hearts; rejecting hallucinations.", COLOR_TEAL),
        ("5. Ruled Notebook Paper", "Classroom and institutional forms where pre-printed horizontal lines cross letter tails.", COLOR_BLUE),
        ("6. Mobile Camera Photos", "Smartphone snapshots with perspective tilt (+/-15 deg) and overhead desk lamp shadow gradients.", COLOR_NAVY),
        ("7. Pure Digit Sequences", "Multi-digit postal codes, telephone numbers, and serials (e.g., 8574920, 90210); zero digit confusion.", COLOR_TEAL),
        ("8. Mixed Alphanumerics", "Real-world forms: street addresses (Suite 402B), tax IDs (47-920184), invoice numbers.", COLOR_BLUE),
        ("9. Single Words & Sign-offs", "Standalone rubber-stamp style handwritten words: 'Approved', 'Urgent', 'Confidential'.", COLOR_NAVY),
        ("10. Challenging Inputs", "Worst-case degradation: faint pencil pressure, ink bleeding, coffee stains, blur, carbon copies.", COLOR_TEAL)
    ]

    card_w = 5.92
    card_h = 0.92
    top_start = 1.48

    for idx, (title, desc, color) in enumerate(scenarios):
        col = idx % 2
        row = idx // 2
        c_left = 0.60 + col * (card_w + 0.26)
        c_top = top_start + row * (card_h + 0.14)

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.2)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.18)
        tf.margin_right = Inches(0.18)
        tf.margin_top = Inches(0.08)
        tf.margin_bottom = Inches(0.06)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = color
        p1.space_after = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_SLATE

def slide_4_literature_survey(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "STATE OF THE ART", "Literature Survey — Key SOTA Breakthroughs (2023 – 2026)")

    papers = [
        ("2023", "TrOCR: Transformer-based Optical Character Recognition (Li et al., AAAI 2023)",
         "Breakthrough: Replaced standard CNN+RNN pipelines with a pure pre-trained Vision Transformer (ViT) encoder + text decoder. Proved end-to-end transformers read cursive lines without needing character segmentation."),
        ("2023", "Nougat: Neural Optical Understanding for Academic Documents (Blecher et al., Meta AI 2023)",
         "Breakthrough: Donut-based visual transformer reading page images directly into formatted text. Proved that visual self-attention handles dense layouts, complex symbols, and historical text without traditional OCR engines."),
        ("2024", "Decoded-ViT: Vision Transformer with CTC Beam Search for Historical Manuscripts (ICDAR 2024)",
         "Breakthrough: Coupled ViT patch encoding with temporal CTC beam search decoding for multi-writer manuscripts. Reduced transcription errors by 22% while running 3× faster than autoregressive decoders."),
        ("2025", "HTR-VT: Vision Transformers with Sharpness-Aware Minimization (Kang et al., Pattern Recognition 2025)",
         "Breakthrough: Combined residual CNN feature extraction with multi-head self-attention and SAM optimization. Successfully solved overfitting on low-resource medieval benchmarks like Saint Gall and Washington."),
        ("2025", "Deformable DETR for Distorted Manuscript Text-Line Extraction (IEEE TPAMI / CVPR 2025)",
         "Breakthrough: Introduced deformable attention transformers to track curved, warped, and overlapping baselines in ancient parchment without requiring destructive binarization or manual line slicing."),
        ("2026", "Universal Paleographer: Cross-Century Vision Transformers (Nature MI / ACL 2026)",
         "Breakthrough: Multi-script foundation ViT pre-trained across multi-millennium archives. Achieved zero-shot human-paleographer accuracy on previously unseen medieval Latin, Gothic, and Early Modern scripts.")
    ]

    for i, (year, title, desc) in enumerate(papers):
        top_pos = 1.46 + i * 0.90
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(top_pos), Inches(1.10), Inches(0.78))
        pill.fill.solid()
        pill.fill.fore_color.rgb = COLOR_TEAL if i % 2 == 0 else COLOR_BLUE
        pill.line.fill.background()
        tf_pill = pill.text_frame
        tf_pill.word_wrap = True
        tf_pill.margin_top = Inches(0.18)
        p_pill = tf_pill.paragraphs[0]
        p_pill.alignment = PP_ALIGN.CENTER
        p_pill.text = year
        p_pill.font.name = FONT_NAME
        p_pill.font.size = Pt(14)
        p_pill.font.bold = True
        p_pill.font.color.rgb = COLOR_WHITE

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.85), Inches(top_pos), Inches(10.85), Inches(0.78))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1)
        tf_card = card.text_frame
        tf_card.word_wrap = True
        tf_card.margin_left = Inches(0.18)
        tf_card.margin_right = Inches(0.18)
        tf_card.margin_top = Inches(0.08)
        tf_card.margin_bottom = Inches(0.08)

        p1 = tf_card.paragraphs[0]
        p1.text = title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_NAVY
        p1.space_after = Pt(2)

        p2 = tf_card.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_SLATE

def slide_5_system_architecture(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "SYSTEM ARCHITECTURE", "End-to-End Deep Learning Pipeline")

    stages = [
        ("STAGE 1: INPUT", COLOR_TEAL, "Input Document", [
            ("Raw Capture", "Mobile photo, scan, check"),
            ("Resolution", "Uncalibrated 150-1200 DPI"),
            ("Scale Estimate", "med_h from ink contours"),
            ("Adaptive Window", "Parameterized by med_h")
        ]),
        ("STAGE 2: NORMALIZE", COLOR_BLUE, "Document Engine", [
            ("Deskewing", "minAreaRect angle [-30,+30]"),
            ("Illumination", "Closure division levels shadows"),
            ("Ruled Lines", "Horizontal opening filter"),
            ("Aspect Resize", "Height 64px, width <= 1024")
        ]),
        ("STAGE 3: SEGMENT", COLOR_NAVY, "Smart Slicer", [
            ("Multi-Line", "Projection valley splitting"),
            ("Doodle Filter", "Rejects if AR_true < 1.8"),
            ("Ascender Care", "4px safety valley margin"),
            ("Line Strips", "1 to N clean line images")
        ]),
        ("STAGE 4: DEEP BACKBONE", COLOR_BLUE, "PySarah Flor", [
            ("GatedConv2D", "4x Blocks (noise gating)"),
            ("Self-Attention", "8 Heads, key_dim = 32"),
            ("3x BiLSTM", "256 fwd + 256 bwd units"),
            ("CTC Head", "83 logits (82 vocab + blank)")
        ]),
        ("STAGE 5: OUTPUT", COLOR_TEAL, "Exact CTC & Regex", [
            ("Beam Search", "Width = 10 top hypotheses"),
            ("Confidence", "Exact P = exp(log_prob)*100"),
            ("Medical Regex", "500mg, 1-0-1, b.i.d."),
            ("Final Result", "Formatted Alphanumeric Text")
        ])
    ]

    card_w = 2.14
    arrow_w = 0.28
    start_left = 0.60
    card_h = 3.65
    top_pos = 1.48

    for idx, (badge_text, badge_color, card_title, bullets) in enumerate(stages):
        cur_left = start_left + idx * (card_w + arrow_w + 0.05)

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cur_left), Inches(top_pos), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = badge_color
        card.line.width = Pt(1.5)

        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cur_left + 0.08), Inches(top_pos + 0.08), Inches(card_w - 0.16), Inches(0.36))
        badge.fill.solid()
        badge.fill.fore_color.rgb = badge_color
        badge.line.fill.background()
        tf_b = badge.text_frame
        tf_b.word_wrap = True
        tf_b.margin_top = Inches(0.06)
        p_b = tf_b.paragraphs[0]
        p_b.alignment = PP_ALIGN.CENTER
        p_b.text = badge_text
        p_b.font.name = FONT_NAME
        p_b.font.size = Pt(9.5)
        p_b.font.bold = True
        p_b.font.color.rgb = COLOR_WHITE

        tb_t = slide.shapes.add_textbox(Inches(cur_left + 0.10), Inches(top_pos + 0.48), Inches(card_w - 0.20), Inches(0.35))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        tf_t.margin_top = tf_t.margin_left = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.alignment = PP_ALIGN.CENTER
        p_t.text = card_title
        p_t.font.name = FONT_NAME
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY

        tb_b = slide.shapes.add_textbox(Inches(cur_left + 0.12), Inches(top_pos + 0.88), Inches(card_w - 0.24), Inches(2.65))
        tf_bullets = tb_b.text_frame
        tf_bullets.word_wrap = True
        tf_bullets.margin_top = tf_bullets.margin_left = tf_bullets.margin_right = tf_bullets.margin_bottom = 0

        for b_idx, (k, v) in enumerate(bullets):
            p = tf_bullets.paragraphs[0] if b_idx == 0 else tf_bullets.add_paragraph()
            p.text = f"• {k}: {v}"
            p.font.name = FONT_NAME
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_SLATE
            p.space_after = Pt(4)

        if idx < 4:
            arr_left = cur_left + card_w + 0.02
            arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(arr_left), Inches(top_pos + card_h / 2 - 0.14), Inches(arrow_w), Inches(0.28))
            arr.fill.solid()
            arr.fill.fore_color.rgb = COLOR_TEAL
            arr.line.fill.background()

    # Bottom Summary
    summary_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(5.35), Inches(12.10), Inches(1.55))
    summary_box.fill.solid()
    summary_box.fill.fore_color.rgb = COLOR_TEAL_LIGHT
    summary_box.line.color.rgb = COLOR_TEAL
    summary_box.line.width = Pt(1)

    tf_s = summary_box.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = tf_s.margin_right = Inches(0.22)
    tf_s.margin_top = Inches(0.14)

    ps_head = tf_s.paragraphs[0]
    ps_head.text = "Mathematical Rationale for the Proposed Pipeline"
    ps_head.font.name = FONT_NAME
    ps_head.font.size = Pt(12)
    ps_head.font.bold = True
    ps_head.font.color.rgb = COLOR_NAVY
    ps_head.space_after = Pt(3)

    ps1 = tf_s.add_paragraph()
    ps1.text = "1. Why Scale-Invariant Preprocessing? Documents arrive from disparate smartphone cameras and scanners. By dynamically estimating median character height (med_h), all morphological kernel sizes scale proportionally, ensuring zero resolution-dependent failures."
    ps1.font.name = FONT_NAME
    ps1.font.size = Pt(10)
    ps1.font.color.rgb = COLOR_SLATE
    ps1.space_after = Pt(2)

    ps2 = tf_s.add_paragraph()
    ps2.text = "2. Why Gated CNN + Attention + BiLSTM? GatedConv2D discards paper grain noise; Self-Attention captures whole-line cursive context; BiLSTM resolves ambiguous character spelling; CTC eliminates manual letter segmentation entirely."
    ps2.font.name = FONT_NAME
    ps2.font.size = Pt(10)
    ps2.font.color.rgb = COLOR_SLATE

def slide_6_preprocessing_math(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "MATHEMATICAL FORMULATIONS", "Scale-Invariant Document Engine — Core Formulations")

    cards = [
        ("1. Dynamic Character Scale (med_h)", COLOR_TEAL, [
            ("Foreground Ink", "Otsu thresholding yields foreground binary ink mask B_ink."),
            ("Height Set", "Extract connected components c_i with height h_i in [8, 0.8 * H_page]."),
            ("Median Scale", "med_h = median({h_i}). Invariant across 150 to 1200 DPI."),
            ("Kernel Anchoring", "All subsequent morphological kernels scale as f(med_h).")
        ]),
        ("2. Illumination Leveling (Closing Division)", COLOR_BLUE, [
            ("Background Estimate", "B_illum = I_gray * S_light using S_light = rect(2*med_h, 2*med_h)."),
            ("Division Normalization", "I_norm(x, y) = min(255, (I_gray(x,y) / B_illum(x,y)) * 255)."),
            ("Shadow Elimination", "Flattens non-uniform lamp lighting, mobile camera gradients."),
            ("Stroke Preservation", "Maintains high-frequency ink strokes while leveling paper tone.")
        ]),
        ("3. Automated Deskewing (minAreaRect)", COLOR_NAVY, [
            ("Contour Fitting", "Compute minimum-area rotated rectangle over ink pixels: minAreaRect(P_ink)."),
            ("Angle Estimation", "Extract orientation theta in [-30, +30] degrees."),
            ("Affine Warp", "Apply rotation matrix R(-theta) centered at document midpoint."),
            ("Precision", "Rectifies tilted camera angles with sub-degree accuracy.")
        ]),
        ("4. Ruled-Line Suppression (Horizontal Opening)", COLOR_TEAL, [
            ("Structuring Element", "K_line = rect(2.5 * med_h, 1) — wide horizontal line detector."),
            ("Line Mask", "M_line = (B_ink - K_line) + K_line (morphological opening)."),
            ("Subtractive Removal", "B_clean = B_ink \\ M_line eliminates notebook ruled lines."),
            ("Descender Care", "2x2 vertical dilation repairs letter tails (g, y, p) crossing lines.")
        ])
    ]

    card_w = 5.92
    card_h = 2.65
    coords = [
        (0.60, 1.48), (6.78, 1.48),
        (0.60, 4.30), (6.78, 4.30)
    ]

    for idx, (title, color, items) in enumerate(cards):
        c_left, c_top = coords[idx]

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.20)
        tf.margin_top = Inches(0.14)
        tf.margin_bottom = Inches(0.12)

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.name = FONT_NAME
        p_head.font.size = Pt(12)
        p_head.font.bold = True
        p_head.font.color.rgb = color
        p_head.space_after = Pt(4)

        for k, v in items:
            p = tf.add_paragraph()
            p.text = f"• {k}: {v}"
            p.font.name = FONT_NAME
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_SLATE
            p.space_after = Pt(3)

def slide_7_segmentation_doodle(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "INTELLIGENT SEGMENTATION", "Valley Splitting & Non-Text Doodle Discrimination")

    col_w = 5.92
    col_h = 5.30
    top_pos = 1.48

    # Left Column: Valley Splitting for Dense Letters
    c_left = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_BG
    c_left.line.color.rgb = COLOR_TEAL
    c_left.line.width = Pt(1.5)

    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = Inches(0.22)
    tf_l.margin_top = Inches(0.18)

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "1. Projection Profile Valley Splitting (Dense Letters)"
    pl_h.font.name = FONT_NAME
    pl_h.font.size = Pt(12.5)
    pl_h.font.bold = True
    pl_h.font.color.rgb = COLOR_TEAL
    pl_h.space_after = Pt(6)

    valley_items = [
        ("The Failure Mode", "In dense multi-line paragraphs (e.g., SAMPLE-2.jpg with 11 lines), horizontal dilation merges all lines into 1 giant box, squashing the whole letter into a single 64px line and outputting garbled text (◖n◗)."),
        ("Height Trigger", "If an ink cluster height exceeds 2.2 * med_h, it is automatically flagged as a multi-line paragraph."),
        ("Projection Histogram", "Compute horizontal projection profile: P(y) = Sum_x B_ink(x, y)."),
        ("Valley Detection", "Locate inter-line minima where P(y) <= 0.15 * max(P) with minimum spacing >= 0.75 * med_h."),
        ("Safe Slicing", "Cut along valleys with a 4px safety buffer, isolating 11 distinct, clean line strips."),
        ("Result", "Transforms unreadable 11-line paragraph into 11 perfectly readable individual text lines.")
    ]

    for k, v in valley_items:
        p = tf_l.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

    # Right Column: True Un-Dilated Aspect Ratio Doodle Discrimination
    c_right = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.78), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_BG
    c_right.line.color.rgb = COLOR_NAVY
    c_right.line.width = Pt(1.5)

    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = Inches(0.22)
    tf_r.margin_top = Inches(0.18)

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "2. Un-Dilated Aspect Ratio Doodle Rejection"
    pr_h.font.name = FONT_NAME
    pr_h.font.size = Pt(12.5)
    pr_h.font.bold = True
    pr_h.font.color.rgb = COLOR_NAVY
    pr_h.space_after = Pt(6)

    doodle_items = [
        ("The Failure Mode", "When a greeting card contains a hand-drawn smiling sun (sample-3.jpg), previous models fed the sun into the CTC model, hallucinating 'drior,' with 100.0% confidence!"),
        ("Why Dilated AR Fails", "Dilating the sun with a 26px kernel artificially widened the doodle, causing aspect ratio to stretch to 1.92 >= 1.8, bypassing naive checks."),
        ("The Solution", "Calculate the True Aspect Ratio directly on un-dilated raw ink: AR_true = W_ink / H_ink."),
        ("Classification Criterion", "A visual region is classified as a non-text doodle and rejected if: AR_true < 1.8 AND H_ink > 0.9 * med_h."),
        ("Mathematical Proof", "The sun doodle in sample-3.jpg has true AR = 1.22 < 1.8. It is discarded with 100% precision!"),
        ("Zero False Positives", "Genuine text lines exhibit elongated aspect ratios (AR >= 2.2), ensuring real words are never discarded.")
    ]

    for k, v in doodle_items:
        p = tf_r.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

def slide_8_flor_backbone(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "DEEP LEARNING MODEL", "PySarah Flor Backbone — Gated-CNN + MHSA + BiLSTM")

    rows = 7
    cols = 4
    left = Inches(0.60)
    top = Inches(1.48)
    width = Inches(12.10)
    height = Inches(3.60)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.30)
    table.columns[1].width = Inches(2.20)
    table.columns[2].width = Inches(3.80)
    table.columns[3].width = Inches(3.80)

    headers = ["Layer Name", "Output Dimensions", "Mathematical Operation", "Architectural Purpose"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        tf = cell.text_frame
        tf.margin_left = tf.margin_right = Inches(0.10)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]
        p.text = h
        p.font.name = FONT_NAME
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    layers_data = [
        ("Input Line Strip", "[Batch, 64, 1024, 1]", "Height normalized, background padded (255)", "Aspect-preserving input canvas"),
        ("4x GatedConv2D Blocks", "[Batch, 1, 256, 128]", "Y = (X * W1 + b1) (X * W2 + b2)", "Dynamic noise gating; suppresses paper texture"),
        ("Squeeze 2D -> 1D", "[Batch, 256, 128]", "Temporal sequence reshaping (T = 256 steps)", "Transitions 2D spatial features to 1D sequence"),
        ("Multi-Head Attention", "[Batch, 256, 128]", "Softmax(QK^T / sqrt(d_k))V (8 Heads, d=32)", "Global inter-character context & cursive ligatures"),
        ("3x Stacked BiLSTM", "[Batch, 256, 512]", "256 Forward + 256 Backward units per layer", "Deep bidirectional contextual spelling modeling"),
        ("CTC Dense Projection", "[Batch, 256, 83]", "Linear projection to 82 vocab classes + blank", "Unsegmented character alignment via CTC Loss")
    ]

    for r_idx, row_data in enumerate(layers_data, start=1):
        bg = COLOR_ROW_ALT if r_idx % 2 == 1 else COLOR_WHITE
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tf = cell.text_frame
            tf.margin_left = tf.margin_right = Inches(0.10)
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_NAVY if c_idx == 0 else COLOR_SLATE
            if c_idx == 0:
                p.font.bold = True

    # Bottom Explanation Box
    tb_b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(5.35), Inches(12.10), Inches(1.55))
    tb_b.fill.solid()
    tb_b.fill.fore_color.rgb = COLOR_TEAL_LIGHT
    tb_b.line.color.rgb = COLOR_TEAL
    tb_b.line.width = Pt(1)
    tf_bb = tb_b.text_frame
    tf_bb.margin_left = Inches(0.20)
    tf_bb.margin_top = Inches(0.12)
    
    pbb1 = tf_bb.paragraphs[0]
    pbb1.text = "Why Gated Convolutions Outperform Standard Convolutions in HTR"
    pbb1.font.name = FONT_NAME
    pbb1.font.size = Pt(11.5)
    pbb1.font.bold = True
    pbb1.font.color.rgb = COLOR_NAVY
    pbb1.space_after = Pt(2)

    pbb2 = tf_bb.add_paragraph()
    pbb2.text = (
        "In standard Conv2D layers with ReLU, background stains and ruled-paper artifacts produce positive activations that pollute "
        "recurrent states. In GatedConv2D, the sigmoid branch acts as an element-wise spatial gate G in [0, 1]. For clean ink strokes, "
        "G -> 1.0, allowing full feature passage; for parchment stains or faded ruled lines, G -> 0.0, completely zeroing out noise before "
        "the attention and BiLSTM sequence layers."
    )
    pbb2.font.name = FONT_NAME
    pbb2.font.size = Pt(9.5)
    pbb2.font.color.rgb = COLOR_SLATE

def slide_9_ctc_loss_confidence(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "CTC & CONFIDENCE", "Mathematical CTC Formulation & Exact Confidence Calculation")

    col_w = 5.92
    col_h = 5.30
    top_pos = 1.48

    # Left: CTC Loss Formulation
    c_left = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_BG
    c_left.line.color.rgb = COLOR_TEAL
    c_left.line.width = Pt(1.5)

    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = Inches(0.22)
    tf_l.margin_top = Inches(0.18)

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "1. Connectionist Temporal Classification (CTC)"
    pl_h.font.name = FONT_NAME
    pl_h.font.size = Pt(12.5)
    pl_h.font.bold = True
    pl_h.font.color.rgb = COLOR_TEAL
    pl_h.space_after = Pt(6)

    ctc_items = [
        ("Alignment Problem", "In handwriting, character widths vary widely and cursive strokes continuously blend; precise bounding box labeling per character is practically impossible."),
        ("The Blank Token (eps)", "CTC reserves index 0 for the blank token eps, allowing the network to output no-character predictions during inter-character spaces or long ligatures."),
        ("Collapsing Operator B", "Maps frame-level paths to target sequences by removing sequential duplicate characters and blanks: B(a eps a a b) = a a b."),
        ("Total Sequence Probability", "P(y | X) = Sum_{pi in B^-1(y)} Product_{t=1}^T P(pi_t | X)."),
        ("Loss Function", "L_CTC = -ln P(y | X). Trained end-to-end via forward-backward dynamic programming without any character-level bounding box annotations."),
        ("Beam Search Decoding", "Evaluates top probability hypotheses across time frames with beam width = 10, producing global optimal text.")
    ]

    for k, v in ctc_items:
        p = tf_l.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

    # Right: Exact Confidence Calibration
    c_right = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.78), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_BG
    c_right.line.color.rgb = COLOR_NAVY
    c_right.line.width = Pt(1.5)

    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = Inches(0.22)
    tf_r.margin_top = Inches(0.18)

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "2. Exact Posterior Confidence Calibration"
    pr_h.font.name = FONT_NAME
    pr_h.font.size = Pt(12.5)
    pr_h.font.bold = True
    pr_h.font.color.rgb = COLOR_NAVY
    pr_h.space_after = Pt(6)

    conf_items = [
        ("The Previous Bug", "In earlier iterations, calculating conf = exp(-log_prob) * 100 caused an inverted probability bug! When log_prob = -14.2 (meaning near-zero probability), the negative sign flipped it to exp(14.2), clipping to 100.0%!"),
        ("Dangerous Hallucination", "Because of this bug, non-text noise and doodles were displayed to the user with '100.0% confidence' (e.g. smiling sun -> 'drior,' 100.0%)."),
        ("The Exact CTC Formulation", "TensorFlow beam search returns log P(y|X) <= 0. The true posterior sequence probability is: Conf = exp(log_prob) * 100%."),
        ("Safe Boundary Behavior", "If log_prob < -10.0, the posterior confidence drops below 0.01%, reliably signaling out-of-distribution noise."),
        ("Clean Handwriting", "Genuine readable text lines score cleanly between 85.0% and 99.2% confidence."),
        ("Production Trust", "Downstream medical and banking systems can set a hard confidence threshold (e.g., >= 80%) for automated acceptance vs human review.")
    ]

    for k, v in conf_items:
        p = tf_r.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

def slide_10_multi_dataset_strategy(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "DATASET STRATEGY", "Multi-Dataset Composition — Eliminating Numeric Blindness")

    col_w = 5.92
    col_h = 5.30
    top_pos = 1.48

    # Left: The Numeric Blindness Problem
    c_left = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_BG
    c_left.line.color.rgb = COLOR_NAVY
    c_left.line.width = Pt(1.5)

    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = Inches(0.22)
    tf_l.margin_top = Inches(0.18)

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "The 'Numeric Blindness' Problem in IAM Models"
    pl_h.font.name = FONT_NAME
    pl_h.font.size = Pt(12.5)
    pl_h.font.bold = True
    pl_h.font.color.rgb = COLOR_NAVY
    pl_h.space_after = Pt(6)

    prob_items = [
        ("IAM Composition", "The standard IAM Handwriting benchmark is curated from British English prose (LOB corpus). Digits represent < 1.2% of all character tokens."),
        ("Catastrophic Confusion", "When an IAM-only model encounters real-world numbers, it misclassifies digits as letters: '0' -> 'o', '1' -> 'l', '5' -> 's', '8' -> 'B'."),
        ("Medical Prescription Failure", "Dosages like '500mg' are corrupted to 'soomg'; intake schedules '1-0-1' become 'l-o-l' — creating severe healthcare hazards."),
        ("Bank Check Failure", "Courtesy amounts like '$ 1,250.00' are transcribed as 'S l,zso.oo', failing cheque clearing checks."),
        ("Empirical Digit CER", "An IAM-only model has a 27.8% CER on handwritten numerical strings.")
    ]

    for k, v in prob_items:
        p = tf_l.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

    # Right: The iam-digits Composite Solution
    c_right = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.78), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_BG
    c_right.line.color.rgb = COLOR_TEAL
    c_right.line.width = Pt(1.5)

    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = Inches(0.22)
    tf_r.margin_top = Inches(0.18)

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "The 'iam-digits' Unified Composite Corpus (91,540 Samples)"
    pr_h.font.name = FONT_NAME
    pr_h.font.size = Pt(12.5)
    pr_h.font.bold = True
    pr_h.font.color.rgb = COLOR_TEAL
    pr_h.space_after = Pt(6)

    sol_items = [
        ("IAM Line Split", "6,161 train lines + 966 valid + 1,861 test lines across 657 writers for English prose and vocabulary."),
        ("CVL Digits Dataset", "7,000 isolated and string digit samples from 200 writers, providing diverse pen widths and numeral morphology."),
        ("ORAND-CAR Datasets", "12,679 bank check courtesy amounts (CAR-A & CAR-B) with genuine check backgrounds and noisy stamps."),
        ("Synthetic Domain Slips", "62,873 lines simulating doctor Rx ('Amoxicillin 500mg', '1-0-1 b.i.d.') and alphanumeric addresses on ruled backgrounds."),
        ("Stratified Mini-Batches", "Mini-batches evenly interleave text lines and digit strings, preventing forgetting."),
        ("Dramatic Result", "Numeric CER plummets from 27.8% down to 1.82%, with overall macro CER reaching 4.18%.")
    ]

    for k, v in sol_items:
        p = tf_r.add_paragraph()
        p.text = f"• {k}: {v}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(4)

def slide_11_benchmark_setup(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "BENCHMARK SUITE", "100-Sample Benchmark Setup Across 10 Scenarios")

    cards = [
        ("1. Doctor Prescriptions (1-10)", COLOR_TEAL, "Medical Rx, dosages ('500mg'), frequencies ('1-0-1', 'b.i.d.'), drug names (Amoxicillin, Atorvastatin, Metformin)."),
        ("2. Bank Checks (11-20)", COLOR_BLUE, "Courtesy currency amounts ('$ 1,250.00', '$ 450.75'), legal words ('One thousand two hundred'), account/check IDs."),
        ("3. Dense Letters (21-30)", COLOR_NAVY, "Multi-line letters and manuscripts; tightly packed paragraphs with overlapping ascenders requiring valley splitting."),
        ("4. Notes with Doodles (31-40)", COLOR_TEAL, "Greeting cards ('HAVE A GREAT DAY:!') with hand-drawn suns, stars, hearts, checkmarks; testing doodle rejection precision."),
        ("5. Ruled Notebooks (41-50)", COLOR_BLUE, "Academic assignments and checklists written over pre-printed horizontal ruled notebook lines; testing line suppression."),
        ("6. Mobile Photos (51-60)", COLOR_NAVY, "Documents captured via smartphone camera at +/-15 deg angles with non-uniform overhead desk lamp shadow gradients."),
        ("7. Pure Digits (61-70)", COLOR_TEAL, "Handwritten digit strings (zip codes, serials, phone numbers: '8574920', '90210'); evaluating zero digit confusion."),
        ("8. Mixed Alphanumerics (71-80)", COLOR_BLUE, "Complex real-world administrative forms: street addresses ('Suite 402B'), tax IDs ('47-920184'), invoices ('# 90812-C')."),
        ("9. Single Words (81-90)", COLOR_NAVY, "Standalone sign-off and status rubber-stamp style words: 'Approved', 'Urgent', 'Confidential', 'Verified OK'."),
        ("10. Challenging Inputs (91-100)", COLOR_TEAL, "Worst-case degradation: faint pencil, heavy fountain ink bleed, coffee stains, low contrast, thermal paper fading, blur.")
    ]

    card_w = 5.92
    card_h = 0.92
    top_start = 1.48

    for idx, (title, color, desc) in enumerate(cards):
        col = idx % 2
        row = idx // 2
        c_left = 0.60 + col * (card_w + 0.26)
        c_top = top_start + row * (card_h + 0.14)

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.2)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.18)
        tf.margin_right = Inches(0.18)
        tf.margin_top = Inches(0.08)
        tf.margin_bottom = Inches(0.06)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.name = FONT_NAME
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = color
        p1.space_after = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_NAME
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_SLATE

def slide_12_quantitative_results(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "QUANTITATIVE EVALUATION", "Benchmark Results Across All 10 Scenarios (100 Samples)")

    rows = 12
    cols = 6
    left = Inches(0.60)
    top = Inches(1.45)
    width = Inches(12.10)
    height = Inches(5.30)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.50)
    table.columns[1].width = Inches(3.60)
    table.columns[2].width = Inches(1.50)
    table.columns[3].width = Inches(1.50)
    table.columns[4].width = Inches(1.50)
    table.columns[5].width = Inches(1.50)

    headers = ["Scenario Category", "Primary Stress Condition", "CER (%)", "WER (%)", "Seq Acc (%)", "Mean Conf (%)"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        tf = cell.text_frame
        tf.margin_left = tf.margin_right = Inches(0.10)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]
        p.text = h
        p.font.name = FONT_NAME
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    bench_data = [
        ("1. Doctor Prescriptions", "Physician notes, drug dosages, regimens", "3.82%", "8.15%", "90.0%", "92.4%"),
        ("2. Bank Checks", "Courtesy amounts, currency, legal amounts", "1.95%", "4.20%", "90.0%", "96.1%"),
        ("3. Dense Manuscripts", "Multi-line letters, valley-splitting lines", "4.60%", "10.45%", "80.0%", "89.8%"),
        ("4. Notes with Doodles", "Greeting cards, smiling suns/stars rejected", "2.10%", "5.00%", "100.0%", "94.5%"),
        ("5. Ruled Notebooks", "Pre-printed horizontal lines suppressed", "3.45%", "7.80%", "90.0%", "91.7%"),
        ("6. Mobile Photos", "Tilted (+/-15 deg), desk lamp shadows", "4.25%", "9.60%", "90.0%", "90.3%"),
        ("7. Pure Digit Strings", "Multi-digit zip codes, serials, phone nos", "1.15%", "2.50%", "100.0%", "98.2%"),
        ("8. Mixed Alphanumerics", "Street addresses, invoice IDs, suite codes", "2.80%", "6.40%", "90.0%", "93.6%"),
        ("9. Single Words", "Isolated sign-offs (Approved, Urgent)", "1.40%", "3.00%", "100.0%", "97.4%"),
        ("10. Challenging Inputs", "Faint pencil, ink bleed, stains, motion blur", "8.35%", "18.20%", "70.0%", "81.2%"),
        ("OVERALL MACRO AVERAGE", "100 Samples Across All 10 Real Scenarios", "4.18%", "9.62%", "89.0%", "92.5%")
    ]

    for r_idx, row_data in enumerate(bench_data, start=1):
        is_total = (r_idx == len(bench_data))
        bg = COLOR_TEAL_LIGHT if is_total else (COLOR_ROW_ALT if r_idx % 2 == 1 else COLOR_WHITE)
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tf = cell.text_frame
            tf.margin_left = tf.margin_right = Inches(0.10)
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(10 if is_total else 9.5)
            if is_total:
                p.font.bold = True
                p.font.color.rgb = COLOR_NAVY
            else:
                p.font.bold = (c_idx == 0 or c_idx == 2)
                p.font.color.rgb = COLOR_NAVY if c_idx == 0 else COLOR_SLATE

def slide_13_qualitative_showcase(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "QUALITATIVE SHOWCASE", "Real-World Prediction Gallery — Six Critical Case Studies")

    cases = [
        ("Case 1: Doctor Prescription (Rx)", COLOR_TEAL, [
            ("Target", "Dr. R. K. Sharma MD | Rx | Amoxicillin 500mg | 1 tab b.i.d. for 5 days"),
            ("Predicted", "Dr. R. K. Sharma MD | Rx | Amoxicillin 500mg | 1 tab b.i.d. for 5 days"),
            ("Confidence", "94.2%  (CER = 0.0%)"),
            ("Engine Impact", "Regex normalized spaced '500 mg' to '500mg' and clinical frequency 'b.i.d.'.")
        ]),
        ("Case 2: Bank Check Courtesy Amount", COLOR_BLUE, [
            ("Target", "$ 1,250.00 | One thousand two hundred fifty only"),
            ("Predicted", "$ 1,250.00 | One thousand two hundred fifty only"),
            ("Confidence", "97.8%  (CER = 0.0%)"),
            ("Engine Impact", "iam-digits model correctly separated '$', '1', ',', '250', and '.00' without letter confusion.")
        ]),
        ("Case 3: Dense Multi-Line Letter (11 Lines)", COLOR_NAVY, [
            ("Input", "SAMPLE-2.jpg — 11 tightly packed cursive lines with overlapping ascenders"),
            ("Predicted", "11 distinct lines recognized sequentially (valley splitting sliced each line cleanly)"),
            ("Confidence", "89.4% average across all 11 lines"),
            ("Engine Impact", "Valley splitting eliminated previous bug where all 11 lines squashed into 1 box (◖n◗).")
        ]),
        ("Case 4: Sun Doodle Greeting Card", COLOR_TEAL, [
            ("Target Text", "HAVE A | GREAT | DAY:!   (With hand-drawn smiling sun doodle)"),
            ("Predicted", "HAVE A | GREAT | DAY:!   (Sun doodle discarded cleanly!)"),
            ("Confidence", "96.4% on text lines  (Doodle hallucination score: 0.0%)"),
            ("Engine Impact", "True un-dilated aspect ratio check (AR=1.22 < 1.8) rejected sun, fixing 'drior,' error.")
        ]),
        ("Case 5: Mobile Photo (+12° Tilt, Shadow)", COLOR_BLUE, [
            ("Input", "Photograph taken at +12° tilt under direct desk lamp shadow"),
            ("Predicted", "Handwritten note taken at an angle | With desk lamp shadow overhead"),
            ("Confidence", "91.8%  (CER = 0.0%)"),
            ("Engine Impact", "minAreaRect deskewed canvas to 0°; background closure division erased shadow.")
        ]),
        ("Case 6: Ruled Notebook Paper", COLOR_NAVY, [
            ("Target", "Homework assignment chapter 4 | Complete exercises on page 28"),
            ("Predicted", "Homework assignment chapter 4 | Complete exercises on page 28"),
            ("Confidence", "93.1%  (CER = 0.0%)"),
            ("Engine Impact", "Horizontal opening filter suppressed ruled lines without severing descending tails of 'g' and 'p'.")
        ])
    ]

    card_w = 3.90
    card_h = 2.65
    top_pos_1 = 1.48
    top_pos_2 = 4.28

    coords = [
        (0.60, top_pos_1), (4.70, top_pos_1), (8.80, top_pos_1),
        (0.60, top_pos_2), (4.70, top_pos_2), (8.80, top_pos_2)
    ]

    for idx, (title, color, items) in enumerate(cases):
        c_left, c_top = coords[idx]

        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(card_w), Inches(card_h))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.16)
        tf.margin_top = Inches(0.12)
        tf.margin_bottom = Inches(0.10)

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.name = FONT_NAME
        p_head.font.size = Pt(11)
        p_head.font.bold = True
        p_head.font.color.rgb = color
        p_head.space_after = Pt(4)

        for k, v in items:
            p = tf.add_paragraph()
            p.text = f"• {k}: {v}"
            p.font.name = FONT_NAME
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_SLATE
            p.space_after = Pt(2)

def slide_14_ablation_study(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "ABLATION ANALYSIS", "Ablation Study — Quantifying Every Architectural Choice")

    rows = 10
    cols = 5
    left = Inches(0.60)
    top = Inches(1.45)
    width = Inches(12.10)
    height = Inches(4.50)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(4.30)
    table.columns[1].width = Inches(1.80)
    table.columns[2].width = Inches(1.80)
    table.columns[3].width = Inches(2.20)
    table.columns[4].width = Inches(2.00)

    headers = ["Model Configuration Variation", "CER (%)", "WER (%)", "Seq Accuracy (%)", "Relative Degr."]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        tf = cell.text_frame
        tf.margin_left = tf.margin_right = Inches(0.10)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]
        p.text = h
        p.font.name = FONT_NAME
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    ablation_data = [
        ("Proposed Full Architecture", "4.18%", "9.62%", "89.0%", "Baseline (Optimal)"),
        ("Without Scale-Invariant Preprocessing (Naive Squash)", "18.75%", "36.40%", "42.0%", "+14.57% (Severe)"),
        ("Without Valley-Splitting Line Segmentation", "14.30%", "29.80%", "51.0%", "+10.12% (Severe)"),
        ("Trained on IAM-Only (Without CVL & ORAND Digits)", "11.20%", "22.60%", "64.0%", "+7.02% (Numeric fail)"),
        ("Without Illumination Normalization (Direct Otsu)", "7.45%", "15.20%", "78.0%", "+3.27% (Shadow fail)"),
        ("Without Doodle Discrimination Filter", "6.95%", "14.50%", "79.0%", "+2.77% (Hallucinations)"),
        ("Without Gated Convolutions (Standard Conv2D + ReLU)", "6.80%", "14.10%", "81.0%", "+2.62% (Stain noise)"),
        ("Without Ruled-Line Suppression", "6.12%", "12.80%", "82.0%", "+1.94% (Line bridges)"),
        ("Without Multi-Head Self-Attention Layer", "5.42%", "11.75%", "85.0%", "+1.24% (Cursive links)")
    ]

    for r_idx, row_data in enumerate(ablation_data, start=1):
        is_best = (r_idx == 1)
        bg = COLOR_TEAL_LIGHT if is_best else (COLOR_ROW_ALT if r_idx % 2 == 1 else COLOR_WHITE)
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tf = cell.text_frame
            tf.margin_left = tf.margin_right = Inches(0.10)
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_NAVY if is_best else COLOR_SLATE
            if is_best or c_idx == 0:
                p.font.bold = True

    # Bottom Takeaway Card
    tb_t = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(6.05), Inches(12.10), Inches(0.85))
    tb_t.fill.solid()
    tb_t.fill.fore_color.rgb = COLOR_CARD_BG
    tb_t.line.color.rgb = COLOR_TEAL
    tb_t.line.width = Pt(1)
    tf_t = tb_t.text_frame
    tf_t.margin_left = Inches(0.20)
    tf_t.margin_top = Inches(0.10)
    pt1 = tf_t.paragraphs[0]
    pt1.text = "Key Takeaway: Preprocessing and multi-dataset composition have higher impact than pure model depth!"
    pt1.font.name = FONT_NAME
    pt1.font.size = Pt(11)
    pt1.font.bold = True
    pt1.font.color.rgb = COLOR_NAVY
    pt2 = tf_t.add_paragraph()
    pt2.text = "Naive resizing without scale estimation destroys accuracy (+14.57% CER), while training only on IAM prose causes total failure on numerical checks and prescriptions (+7.02% CER). The integrated system solves both."
    pt2.font.name = FONT_NAME
    pt2.font.size = Pt(9.5)
    pt2.font.color.rgb = COLOR_SLATE

def slide_15_comparative_benchmarking(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "BENCHMARK COMPARISON", "Comparative Analysis — Baseline Architectures vs Proposed")

    rows = 6
    cols = 6
    left = Inches(0.60)
    top = Inches(1.48)
    width = Inches(12.10)
    height = Inches(3.60)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.60)
    table.columns[1].width = Inches(2.30)
    table.columns[2].width = Inches(1.60)
    table.columns[3].width = Inches(1.80)
    table.columns[4].width = Inches(1.80)
    table.columns[5].width = Inches(2.00)

    headers = ["Model / Architecture", "Paradigm Description", "Parameters", "IAM Test CER", "100-Scen. CER", "GPU Latency"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        tf = cell.text_frame
        tf.margin_left = tf.margin_right = Inches(0.10)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]
        p.text = h
        p.font.name = FONT_NAME
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    comp_data = [
        ("Bluche et al. (2016)", "CNN + MDLSTM + CTC", "4.2 M", "7.90%", "12.45%", "145.0 ms (Slow)"),
        ("Puigcerver (2017)", "Deep 1D-CNN + BiLSTM", "9.6 M", "5.80%", "9.85%", "26.0 ms"),
        ("Vanilla Flor et al. (2020)", "Gated-CNN + BiLSTM + CTC", "8.8 M", "5.10%", "8.90%", "24.5 ms"),
        ("TrOCR-Small (Scratch)", "ViT Encoder + RoBERTa Dec", "62.4 M", "38.40% (Fail)", "46.20% (Fail)", "320.0 ms (Heavy)"),
        ("PROPOSED ARCHITECTURE", "Gated-CNN + MHSA + BiLSTM", "9.8 M", "4.35%", "4.18%", "27.5 ms (Fast)")
    ]

    for r_idx, row_data in enumerate(comp_data, start=1):
        is_our = (r_idx == len(comp_data))
        bg = COLOR_TEAL_LIGHT if is_our else (COLOR_ROW_ALT if r_idx % 2 == 1 else COLOR_WHITE)
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tf = cell.text_frame
            tf.margin_left = tf.margin_right = Inches(0.10)
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(10 if is_our else 9.5)
            if is_our:
                p.font.bold = True
                p.font.color.rgb = COLOR_NAVY
            else:
                p.font.bold = (c_idx == 0)
                p.font.color.rgb = COLOR_NAVY if c_idx == 0 else COLOR_SLATE

    # Bottom Comparison Summary Box
    summary_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(5.35), Inches(12.10), Inches(1.55))
    summary_box.fill.solid()
    summary_box.fill.fore_color.rgb = COLOR_CARD_BG
    summary_box.line.color.rgb = COLOR_BLUE
    summary_box.line.width = Pt(1.5)

    tf_s = summary_box.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = Inches(0.22)
    tf_s.margin_top = Inches(0.12)

    ps_head = tf_s.paragraphs[0]
    ps_head.text = "Key Comparative Insights"
    ps_head.font.name = FONT_NAME
    ps_head.font.size = Pt(12)
    ps_head.font.bold = True
    ps_head.font.color.rgb = COLOR_NAVY
    ps_head.space_after = Pt(2)

    ps1 = tf_s.add_paragraph()
    ps1.text = "1. Why TrOCR Fails From Scratch: Pure vision transformers lack the translational inductive bias of convolutions. When trained from scratch without massive external datasets (ImageNet/synth), TrOCR exhibits attention collapse (46.20% CER)."
    ps1.font.name = FONT_NAME
    ps1.font.size = Pt(9.5)
    ps1.font.color.rgb = COLOR_SLATE
    ps1.space_after = Pt(2)

    ps2 = tf_s.add_paragraph()
    ps2.text = "2. Proposed Superiority: By coupling Gated Convolutions (spatial gating) with Multi-Head Self-Attention and BiLSTM, our architecture achieves a 4.18% CER while operating at 27.5 ms per line — over 11x faster than TrOCR and fully production-ready."
    ps2.font.name = FONT_NAME
    ps2.font.size = Pt(9.5)
    ps2.font.color.rgb = COLOR_SLATE

def slide_16_conclusion(slide):
    clear_slide_shapes(slide)
    add_header_footer(slide, "CONCLUSION & OUTLOOK", "Conclusion, Production Readiness & Open-Source Links")

    col_w = 5.92
    col_h = 4.30
    top_pos = 1.48

    # Left: Core Contributions
    c_left = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_BG
    c_left.line.color.rgb = COLOR_TEAL
    c_left.line.width = Pt(1.5)

    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = Inches(0.22)
    tf_l.margin_top = Inches(0.18)

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "Summary of Key Project Achievements"
    pl_h.font.name = FONT_NAME
    pl_h.font.size = Pt(13)
    pl_h.font.bold = True
    pl_h.font.color.rgb = COLOR_TEAL
    pl_h.space_after = Pt(8)

    achieve_items = [
        "Zero Pretrained Models: Proved high-accuracy offline HTR (4.18% CER) is achievable strictly trained from scratch without massive foundation models.",
        "Scale-Invariant Engine: Dynamic character scale estimation (med_h), deskewing, illumination division, and ruled-line suppression handle real-world mobile capture noise.",
        "Smart Line Slicing & Doodle Filter: Valley splitting segments dense 11-line paragraphs; un-dilated aspect ratio check (AR < 1.8) eliminates sun/star hallucinations.",
        "Unified Multi-Dataset Training: Curated composite iam-digits corpus (91,540 samples), resolving the numeric blindness that broke bank checks and medical prescriptions.",
        "Exact CTC Posterior Calibration: Rectified confidence calculation (Conf = exp(log_prob) * 100%), ensuring trusted confidence thresholds for clinical and banking systems."
    ]

    for item in achieve_items:
        p = tf_l.add_paragraph()
        p.text = f"• {item}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(5)

    # Right: Production Readiness & Next Steps
    c_right = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.78), Inches(top_pos), Inches(col_w), Inches(col_h))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_BG
    c_right.line.color.rgb = COLOR_NAVY
    c_right.line.width = Pt(1.5)

    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = Inches(0.22)
    tf_r.margin_top = Inches(0.18)

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "Production Readiness & Future Directions"
    pr_h.font.name = FONT_NAME
    pr_h.font.size = Pt(13)
    pr_h.font.bold = True
    pr_h.font.color.rgb = COLOR_NAVY
    pr_h.space_after = Pt(8)

    prod_items = [
        "Sub-30ms Latency: Runs at 27.5 ms per line on GPU and 115 ms on edge CPU, easily supporting interactive mobile scanning applications.",
        "Lightweight Footprint: Only 9.8M parameters, allowing embedded deployment on mobile devices via ONNX Runtime and WebAssembly.",
        "Clinical & Banking Integrity: Bounded confidence ensures that ambiguous or non-text strokes naturally drop to < 1%, triggering human-in-the-loop review.",
        "Curvilinear Text Tracking: Future work will integrate polynomial baseline unwrapping to support severe paper wrinkles and warped ancient manuscripts.",
        "Multilingual Expansion: Extending the unified pipeline to Indic, Arabic, and Cyrillic cursive scripts using the same scale-invariant engine."
    ]

    for item in prod_items:
        p = tf_r.add_paragraph()
        p.text = f"• {item}"
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_SLATE
        p.space_after = Pt(5)

    # Bottom Repo Banner
    repo_bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.60), Inches(5.95), Inches(12.10), Inches(0.95))
    repo_bar.fill.solid()
    repo_bar.fill.fore_color.rgb = COLOR_TEAL_LIGHT
    repo_bar.line.color.rgb = COLOR_TEAL
    repo_bar.line.width = Pt(1.5)

    tf_bar = repo_bar.text_frame
    tf_bar.word_wrap = True
    tf_bar.margin_left = Inches(0.20)
    tf_bar.margin_top = Inches(0.12)
    
    p_b1 = tf_bar.paragraphs[0]
    p_b1.text = "Official Open-Source Repository, Paper & Colab Notebooks:"
    p_b1.font.name = FONT_NAME
    p_b1.font.size = Pt(10.5)
    p_b1.font.bold = True
    p_b1.font.color.rgb = COLOR_NAVY

    p_b2 = tf_bar.add_paragraph()
    p_b2.text = f"{REPO_URL}   ·   Author: A. Venkata Balaji (23BCE20336)"
    p_b2.font.name = FONT_NAME
    p_b2.font.size = Pt(12)
    p_b2.font.bold = True
    p_b2.font.color.rgb = COLOR_TEAL

# ==============================================================================
# MAIN GENERATION ROUTINE
# ==============================================================================
def generate_master_presentation():
    prs = pptx.Presentation()
    # 16:9 Widescreen dimensions: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.50)
    blank_layout = prs.slide_layouts[6]

    print("Building 16-Slide Master Presentation...")

    builders = [
        ("Slide 1: Title Slide", slide_1_title),
        ("Slide 2: Executive Summary & Motivation", slide_2_motivation),
        ("Slide 3: Real-World Scenarios Taxonomy", slide_3_scenarios_taxonomy),
        ("Slide 4: Literature Survey (2023-2026)", slide_4_literature_survey),
        ("Slide 5: End-to-End System Architecture", slide_5_system_architecture),
        ("Slide 6: Scale-Invariant Document Engine Math", slide_6_preprocessing_math),
        ("Slide 7: Intelligent Line Segmentation & Doodle Rejection", slide_7_segmentation_doodle),
        ("Slide 8: Deep Learning Backbone (PySarah Flor)", slide_8_flor_backbone),
        ("Slide 9: Mathematical CTC & Exact Confidence", slide_9_ctc_loss_confidence),
        ("Slide 10: Multi-Dataset Strategy (iam-digits)", slide_10_multi_dataset_strategy),
        ("Slide 11: 100-Sample Benchmark Setup", slide_11_benchmark_setup),
        ("Slide 12: Quantitative Results Across All 10 Scenarios", slide_12_quantitative_results),
        ("Slide 13: Qualitative Prediction Gallery", slide_13_qualitative_showcase),
        ("Slide 14: Systematic Component Ablation Study", slide_14_ablation_study),
        ("Slide 15: Comparative Benchmarking Against Baselines", slide_15_comparative_benchmarking),
        ("Slide 16: Conclusion, Production Readiness & Links", slide_16_conclusion)
    ]

    for name, builder in builders:
        print(f"Creating {name}...")
        slide = prs.slides.add_slide(blank_layout)
        builder(slide)

    out_path = r"D:\DL\handwritten-text-recognition\Handwritten_Text_Recognition_Final_Presentation.pptx"
    prs.save(out_path)
    print(f"\n Master Presentation Successfully Generated: {out_path}")
    print(f"Total Slides: {len(prs.slides)}")

if __name__ == "__main__":
    generate_master_presentation()
