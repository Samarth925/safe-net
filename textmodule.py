import io
from xml.sax.saxutils import escape
from fastapi.responses import Response
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detoxify import Detoxify

app = FastAPI(title="Safe-net API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pretrained AI model
model = Detoxify("original")

# Temporary in-memory history for the demo
history = []


@app.get("/")
def home():
    return {"message": "Safe-net API is running"}


@app.post("/send")
def send(msg: str):
    text = msg.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Analyze the message with AI
    scores = model.predict(text)
    category_scores = {
        category: float(score)
        for category, score in scores.items()
    }

    # Basic repeated-message signal
    normalized = text.lower()
    recent = history[-10:]
    repeats = recent.count(normalized)
    history.append(normalized)

    # Combine AI score with a simple repetition adjustment
    ai_score = max(category_scores.values(), default=0.0)
    base_risk = round(ai_score * 100)
    repeat_boost = min(30, repeats * 10)
    risk = min(100, base_risk + repeat_boost)

    if risk >= 70:
        action = "flag_for_review"
    elif risk >= 35:
        action = "warn"
    else:
        action = "allow"

    return {
        "message": text,
        "model": "Detoxify original",
        "risk": risk,
        "category_scores": {
            key: round(value, 3)
            for key, value in category_scores.items()
        },
        "repeated_count": repeats + 1,
        "action": action
    }

class MessageRequest(BaseModel):
    sender_id: str
    receiver_id: str
    msg: str
    timestamp: datetime | None = None

structured_messages = []

@app.post("/message")
def message(req: MessageRequest):
    text = req.msg.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    ts = req.timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    scores = {k: float(v) for k, v in model.predict(text).items()}
    toxicity = max(scores.values(), default=0.0) * 100

    structured_messages.append({
        "sender": req.sender_id, "receiver": req.receiver_id,
        "ts": ts, "toxicity": toxicity, "text": text,
    })

    pair = [m for m in structured_messages
            if m["sender"] == req.sender_id and m["receiver"] == req.receiver_id]

    recent = [m for m in pair if m["ts"] >= ts - timedelta(minutes=10)]
    frequency = min(len(recent) * 8, 25)

    late_night = 15 if ts.hour < 5 else 0

    tox = [m["toxicity"] for m in pair]
    escalation = 0
    if len(tox) >= 4:
        last, prev = tox[-3:], tox[-6:-3] or tox[:-3]
        if sum(last)/len(last) > sum(prev)/len(prev) + 5:
            escalation = 15

    risk = min(100, round(toxicity * 0.5 + frequency + late_night + escalation))

    action = "flag_for_review" if risk >= 70 else "warn" if risk >= 35 else "allow"
    structured_messages[-1]["risk"] = risk
    structured_messages[-1]["action"] = action

    return {"risk": risk, "action": action,
            "breakdown": {"toxicity": round(toxicity, 1), "frequency": frequency,
                          "late_night": late_night, "escalation": escalation},
            "category_scores": {k: round(v, 3) for k, v in scores.items()}}

@app.get("/report/{sender_id}/{receiver_id}")
def report(sender_id: str, receiver_id: str):
    pair = [m for m in structured_messages
            if m["sender"] == sender_id and m["receiver"] == receiver_id]
    if not pair:
        raise HTTPException(status_code=404, detail="No messages for this pair")

    def entry(m):
        return {"time": m["ts"].isoformat(), "message": m["text"],
                "toxicity": round(m["toxicity"], 1),
                "risk": m.get("risk"), "action": m.get("action")}

    flagged = [m for m in pair if m.get("risk", 0) >= 35]
    peak = max(m.get("risk", 0) for m in pair)
    level = "high" if peak >= 70 else "medium" if peak >= 35 else "low"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sender": sender_id,
        "receiver": receiver_id,
        "summary": {
            "risk_level": level,
            "peak_risk": peak,
            "total_messages": len(pair),
            "flagged_messages": len(flagged),
            "avg_toxicity": round(sum(m["toxicity"] for m in pair) / len(pair), 1),
            "first_message": pair[0]["ts"].isoformat(),
            "last_message": pair[-1]["ts"].isoformat(),
        },
        "timeline": [entry(m) for m in pair],
        "flagged": [entry(m) for m in flagged],
    }
@app.get("/report/{sender_id}/{receiver_id}/pdf")
def report_pdf(sender_id: str, receiver_id: str):
    data = report(sender_id, receiver_id)
    s = data["summary"]
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Sentinel Evidence Report", styles["Title"]),
        Paragraph(f"Generated: {data['generated_at'][:19].replace('T', ' ')} UTC", styles["Normal"]),
        Paragraph(f"Sender: {escape(sender_id)} &nbsp;|&nbsp; Receiver: {escape(receiver_id)}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Risk level: <b>{s['risk_level'].upper()}</b> (peak {s['peak_risk']}/100)", styles["Heading2"]),
        Paragraph(f"Messages: {s['total_messages']} &nbsp;|&nbsp; Flagged: {s['flagged_messages']} &nbsp;|&nbsp; Avg toxicity: {s['avg_toxicity']}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Timeline", styles["Heading2"]),
    ]

    rows = [["Time (UTC)", "Message", "Tox", "Risk", "Action"]]
    for e in data["timeline"]:
        rows.append([
            e["time"][:19].replace("T", " "),
            Paragraph(escape(e["message"]), styles["Normal"]),
            e["toxicity"], e["risk"], e["action"],
        ])

    table = Table(rows, colWidths=[95, 200, 40, 40, 95], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i, e in enumerate(data["timeline"], start=1):
        if (e["risk"] or 0) >= 70:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.mistyrose))
    table.setStyle(TableStyle(style))
    story.append(table)

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="report_{sender_id}_{receiver_id}.pdf"'},
    )
