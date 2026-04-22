import re
from typing import Dict, Iterable, Optional

import pandas as pd

RATING_COLUMNS = ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5", "rating_6"]
DEPARTMENT_PATTERNS: Dict[str, Iterable[str]] = {
    "Mathematics": ["ریاضی", "جبر", "آمار"],
    "Physics": ["فیزیک", "الکترومغناطیس", "مکانیک"],
    "Computer Engineering": ["کامپیوتر", "برنامه", "الگوریتم", "داده"],
    "Electrical Engineering": ["برق", "مدار", "الکترونیک", "کنترل"],
    "Civil Engineering": ["عمران", "سازه", "نقشه"],
    "Industrial Engineering": ["صنایع", "بهینه", "تحقیق در عملیات"],
}

COMMENT_NOISE_PATTERNS = [
    r"~{3,}",
    r"برای ثبت معرفی استاد به ربات زیر پیام بدید",
    r"کانال معرفی اساتید دانشگاه علم و صنعت",
    r"@ostad_elmosiBot",
    r"@ostad_elmosi",
]


def _normalize_persian_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    normalized = str(text).replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ").strip()
    return re.sub(r"\s+", " ", normalized)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    return value_str in {"true", "1", "yes"}


def extract_department(course_name: object, fallback_department: object = None) -> Optional[str]:
    course_text = _normalize_persian_text(course_name)
    fallback_text = _normalize_persian_text(fallback_department)
    haystack = f"{course_text} {fallback_text}".strip()
    if not haystack:
        return None
    for department, patterns in DEPARTMENT_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return department
    return fallback_text if fallback_text else None


def standardize_grading(text: object) -> str:
    value = _normalize_persian_text(text)
    if not value:
        return "unknown"
    if any(key in value for key in ["منصفانه", "عادلانه"]):
        return "fair"
    if any(key in value for key in ["نمره خوبی نمیشه", "سخت گیر", "به سختی نمره"]):
        return "strict"
    if any(key in value for key in ["دست باز", "ارفاق", "راحت نمره"]):
        return "lenient"
    return "unknown"


def standardize_attendance(text: object) -> str:
    value = _normalize_persian_text(text)
    if not value:
        return "unknown"
    if any(key in value for key in ["حضور مهم است", "تاثیر مستقیم دارد", "اجباری"]):
        return "mandatory"
    if "حضور مهم نیست" in value and "تاثیر مثبت" in value:
        return "positive_effect"
    if any(key in value for key in ["حضور و غیاب نمی کند", "اختیاری", "حضور مهم نیست"]):
        return "optional"
    return "unknown"


def _clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            if column == "comment_text":
                result[column] = result[column].map(clean_comment_text)
            else:
                result[column] = result[column].map(_normalize_persian_text)
            result[column] = result[column].replace("", pd.NA)
    return result


def clean_comment_text(text: object) -> str:
    value = _normalize_persian_text(text)
    if not value:
        return ""
    for pattern in COMMENT_NOISE_PATTERNS:
        value = re.sub(pattern, " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _compute_rating_mean(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    usable_rating_cols = [column for column in RATING_COLUMNS if column in result.columns]
    if not usable_rating_cols:
        result["rating_mean"] = pd.NA
        return result
    for column in usable_rating_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower=0, upper=10)
    result["rating_mean"] = result[usable_rating_cols].mean(axis=1, skipna=True)
    return result


def clean_reviews_dataframe(
    df: pd.DataFrame,
    drop_parse_errors: bool = True,
    drop_empty_comments: bool = True,
) -> pd.DataFrame:
    result = df.copy()

    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if "date_unixtime" in result.columns:
        result["date_unixtime"] = pd.to_numeric(result["date_unixtime"], errors="coerce")

    result = _clean_text_columns(
        result,
        [
            "professor_name_raw",
            "department",
            "course_name",
            "grading_status_raw",
            "attendance_status_raw",
            "comment_text",
            "term",
        ],
    )

    if drop_parse_errors and "parse_error" in result.columns:
        parse_error_series = result["parse_error"].map(_to_bool)
        result = result.loc[~parse_error_series]

    result = _compute_rating_mean(result)

    if "course_name" in result.columns or "department" in result.columns:
        course_series = result["course_name"] if "course_name" in result.columns else pd.Series([None] * len(result))
        department_series = result["department"] if "department" in result.columns else pd.Series([None] * len(result))
        result["department_std"] = [
            extract_department(course_value, department_value)
            for course_value, department_value in zip(course_series, department_series)
        ]

    if "grading_status_raw" in result.columns:
        result["grading_status_std"] = result["grading_status_raw"].map(standardize_grading)
    else:
        result["grading_status_std"] = "unknown"

    if "attendance_status_raw" in result.columns:
        result["attendance_status_std"] = result["attendance_status_raw"].map(standardize_attendance)
    else:
        result["attendance_status_std"] = "unknown"

    if drop_empty_comments and "comment_text" in result.columns:
        result = result[result["comment_text"].notna() & (result["comment_text"].str.len() > 2)]

    if "professor_name_raw" in result.columns:
        result = result[result["professor_name_raw"].notna()]

    return result.reset_index(drop=True)


def load_and_clean_reviews(csv_path: str, **kwargs) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    return clean_reviews_dataframe(raw_df, **kwargs)

