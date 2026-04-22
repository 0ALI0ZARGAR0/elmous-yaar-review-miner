import re
from typing import Dict


def normalize_text(x) -> str:
    """
    Convert heterogeneous Telegram text (list / dict / string)
    into a single raw string for parsing.
    """
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(x)


def parse_message(raw_text: str) -> Dict:
    parsed = {
        "professor_id": None,
        "professor_name_raw": None,
        "department": None,
        "course_name": None,
        "rating_1": None,
        "rating_2": None,
        "rating_3": None,
        "rating_4": None,
        "rating_5": None,
        "rating_6": None,
        "grading_status_raw": None,
        "attendance_status_raw": None,
        "comment_text": None,
        "term": None,
        "parse_error": False
    }

    if not raw_text or len(raw_text.strip()) < 20:
        parsed["parse_error"] = True
        return parsed

    text = raw_text.replace("\u200c", " ").strip()

    # Professor name
    name_match = re.search(r"🧑‍🏫\s*([^\n]+)", text)
    if name_match:
        parsed["professor_name_raw"] = name_match.group(1).strip()

    # Course name
    course_match = re.search(r"📒\s*([^\n]+)", text)
    if course_match:
        parsed["course_name"] = course_match.group(1).strip()

    # Department
    dept_match = re.search(r"#([^\s#]+)", text)
    if dept_match:
        parsed["department"] = dept_match.group(1)

    # Ratings
    rating_patterns = {
        "rating_1": r"پیوستگی.*?:\s*(\d+)",
        "rating_2": r"دانش عمومی.*?:\s*(\d+)",
        "rating_3": r"انتقال مطالب.*?:\s*(\d+)",
        "rating_4": r"مدیریت کلاس.*?:\s*(\d+)",
        "rating_5": r"پاسخگویی.*?:\s*(\d+)",
        "rating_6": r"آداب و رفتار.*?:\s*(\d+)"
    }

    for key, pattern in rating_patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = int(match.group(1))

    # Grading
    grading_match = re.search(r"وضعیت نمره دادن:\s*┘\s*([^\n]+)", text)
    if grading_match:
        parsed["grading_status_raw"] = grading_match.group(1).strip()

    # Attendance
    attendance_match = re.search(r"حضور و غیاب\s*┘\s*([^\n]+)", text)
    if attendance_match:
        parsed["attendance_status_raw"] = attendance_match.group(1).strip()

    # Term
    term_match = re.search(r"ترمی که.*?:\s*┘\s*([^\n]+)", text)
    if term_match:
        parsed["term"] = term_match.group(1).strip()

    # Comment
    comment_match = re.search(r"توضیحات:\s*┘([\s\S]+)", text)
    if comment_match:
        parsed["comment_text"] = comment_match.group(1).strip()

    # Validation
    if parsed["professor_name_raw"] is None or parsed["course_name"] is None:
        parsed["parse_error"] = True

    return parsed
