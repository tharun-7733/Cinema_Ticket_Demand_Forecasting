import csv
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

app = FastAPI(title="CinemaOps – Ticket Demand Forecasting")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = Path("artifacts/cinema_random_forest.pkl")
METRICS_PATH = Path("report/model_evaluation_metrics.csv")
MODEL_VERSION = "cinema-rf-v1.0"

rf_pipeline = joblib.load(MODEL_PATH)
preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_estimator = rf_pipeline.named_steps["model"]
num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
cat_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]

NUM_FEATURES = [
    "capacity",
    "ticket_price",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
]
CAT_FEATURES = ["film_code", "cinema_code", "show_time"]
FEATURES = [
    "film_code",
    "cinema_code",
    "show_time",
    "capacity",
    "ticket_price",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
]

TRAINED_FILMS = sorted(int(x) for x in ohe.categories_[0])
TRAINED_CINEMAS = sorted(int(x) for x in ohe.categories_[1])
TRAINED_SHOWTIMES = sorted(int(x) for x in ohe.categories_[2])
TRAINED_FILM_SET = set(TRAINED_FILMS)
TRAINED_CINEMA_SET = set(TRAINED_CINEMAS)

REF_VALUES = {
    **dict(zip(NUM_FEATURES, [float(v) for v in num_imputer.statistics_])),
    **dict(zip(CAT_FEATURES, [int(v) for v in cat_imputer.statistics_])),
}

FEATURE_LABELS = {
    "capacity": "Hall capacity",
    "ticket_price": "Ticket price",
    "show_time": "Shows / slot",
    "film_code": "Movie",
    "cinema_code": "Screen / cinema",
    "day_of_week": "Weekday",
    "day_of_month": "Day of month",
    "month": "Month",
    "quarter": "Quarter",
    "is_weekend": "Weekend flag",
}


class ForecastRequest(BaseModel):
    film_code: int = Field(..., ge=1)
    cinema_code: int = Field(..., ge=1)
    show_time: int = Field(..., ge=1, le=60)
    capacity: float = Field(..., gt=0)
    ticket_price: float = Field(..., gt=0)
    forecast_date: str
    days: int = Field(7, ge=1, le=30)


class PredictRequest(BaseModel):
    film_code: int = Field(..., ge=1)
    cinema_code: int = Field(..., ge=1)
    show_time: int = Field(..., ge=1, le=60)
    capacity: float = Field(..., gt=0)
    ticket_price: float = Field(..., gt=0)
    forecast_date: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=120)


def make_feature_dict(req: PredictRequest, dt: date) -> dict:
    return {
        "film_code": req.film_code,
        "cinema_code": req.cinema_code,
        "show_time": req.show_time,
        "capacity": req.capacity,
        "ticket_price": req.ticket_price,
        "day_of_week": dt.weekday(),
        "day_of_month": dt.day,
        "month": dt.month,
        "quarter": (dt.month - 1) // 3 + 1,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
    }


def food_plan(tickets: int, occ: float) -> dict:
    if occ >= 0.75:
        rates = (0.75, 0.90, 0.40, 0.35)
        level = "HIGH"
    elif occ >= 0.45:
        rates = (0.60, 0.75, 0.30, 0.22)
        level = "MEDIUM"
    else:
        rates = (0.40, 0.55, 0.18, 0.12)
        level = "LOW"
    return {
        "level": level,
        "popcorn": round(tickets * rates[0]),
        "drinks": round(tickets * rates[1]),
        "snacks": round(tickets * rates[2]),
        "combos": round(tickets * rates[3]),
    }


def schedule_plan(occ: float, show_time: int) -> dict:
    if occ >= 0.75:
        return {
            "shows": min(show_time + 1, 14),
            "staff": "Full + 2 extra",
            "action": "Consider adding a show, opening all counters, and preparing waitlist",
        }
    if occ >= 0.45:
        return {
            "shows": show_time,
            "staff": "Standard staff",
            "action": "Keep standard show schedule and monitor walk-in pacing",
        }
    return {
        "shows": max(show_time - 1, 1),
        "staff": "Lean team",
        "action": "Consider shifting to a smaller screen or offering off-peak bundles",
    }


def marketing_tip(occ_pct: float) -> str:
    if occ_pct < 40:
        return "Flash discount (20-30%) or bundle deal for walk-ins"
    if occ_pct < 65:
        return "Push combo upsells at checkout & social media"
    if occ_pct < 85:
        return "Upsell premium seats & loyalty rewards"
    return "Near sellout! Enable waitlist & promote next showtime"


def run_single_prediction(req: PredictRequest, dt: date, include_contributions: bool = True) -> dict:
    t0 = time.perf_counter()
    row = make_feature_dict(req, dt)
    df = pd.DataFrame([row])[FEATURES]
    X_t = preprocessor.transform(df)

    tree_preds = np.array([tree.predict(X_t)[0] for tree in rf_estimator.estimators_])
    raw_mean = float(tree_preds.mean())
    p10 = float(np.percentile(tree_preds, 10))
    p90 = float(np.percentile(tree_preds, 90))

    cap = float(req.capacity)
    pred_tickets = min(round(cap), max(0, round(raw_mean)))
    low_tickets = min(round(cap), max(0, round(p10)))
    high_tickets = min(round(cap), max(low_tickets, round(p90)))
    occ_pct = round(pred_tickets / cap * 100, 1)

    # Median-reference baseline scaled to requested hall capacity
    ref_df = pd.DataFrame([REF_VALUES])[FEATURES]
    ref_pred = float(rf_pipeline.predict(ref_df)[0])
    baseline_pred = min(round(cap), max(0, round(ref_pred * (cap / REF_VALUES["capacity"]))))

    cold_film = req.film_code not in TRAINED_FILM_SET
    cold_cinema = req.cinema_code not in TRAINED_CINEMA_SET
    cold_start = cold_film or cold_cinema

    top_features = []
    explanation = "Prediction reflects historical demand patterns for similar screen and pricing parameters."
    if include_contributions:
        perturb_rows = []
        keys = ["capacity", "ticket_price", "show_time", "film_code", "cinema_code", "day_of_week", "is_weekend", "month"]
        for k in keys:
            r_copy = dict(row)
            r_copy[k] = REF_VALUES[k]
            perturb_rows.append(r_copy)
        perturb_df = pd.DataFrame(perturb_rows)[FEATURES]
        perturb_preds = rf_pipeline.predict(perturb_df)
        contribs = [
            {"name": FEATURE_LABELS[k], "contribution": round(raw_mean - float(p), 1)}
            for k, p in zip(keys, perturb_preds)
        ]
        contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        top_features = contribs[:6]

        pos = [c for c in top_features if c["contribution"] > 0]
        neg = [c for c in top_features if c["contribution"] < 0]
        parts = []
        if pos:
            parts.append(f"{pos[0]['name']} (+{round(pos[0]['contribution'])} tickets) lifts demand")
        if neg:
            parts.append(f"{neg[0]['name']} ({round(neg[0]['contribution'])} tickets) pulls it down")
        if parts:
            explanation = "; ".join(parts) + " relative to a typical baseline show."

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    occ_ratio = pred_tickets / cap

    return {
        "film_code": req.film_code,
        "cinema_code": req.cinema_code,
        "show_time": req.show_time,
        "forecast_date": dt.isoformat(),
        "day": dt.strftime("%A"),
        "predicted_tickets": pred_tickets,
        "predicted_occupancy": occ_pct,
        "interval_low": low_tickets,
        "interval_high": high_tickets,
        "capacity": cap,
        "baseline_prediction": baseline_pred,
        "cold_start": cold_start,
        "history_count": 0 if cold_start else 1,
        "top_features": top_features,
        "explanation": explanation,
        "suggested_action": schedule_plan(occ_ratio, req.show_time)["action"],
        "food": food_plan(pred_tickets, occ_ratio),
        "schedule": schedule_plan(occ_ratio, req.show_time),
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"cache_bust": int(time.time())})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "n_estimators": rf_estimator.n_estimators,
    }


@app.get("/options")
async def options():
    return {
        "movies": TRAINED_FILMS,
        "screens": TRAINED_CINEMAS,
        "showtimes": TRAINED_SHOWTIMES,
        "defaults": {
            "film_code": int(REF_VALUES["film_code"]),
            "cinema_code": int(REF_VALUES["cinema_code"]),
            "show_time": int(REF_VALUES["show_time"]),
            "capacity": round(REF_VALUES["capacity"]),
            "ticket_price": round(REF_VALUES["ticket_price"]),
        },
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        dt = date.fromisoformat(req.forecast_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc
    return run_single_prediction(req, dt, include_contributions=True)


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    t0 = time.perf_counter()
    results = []
    for item in req.items:
        try:
            dt = date.fromisoformat(item.forecast_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date: {item.forecast_date}") from exc
        results.append(run_single_prediction(item, dt, include_contributions=True))
    return {
        "predictions": results,
        "model_version": MODEL_VERSION,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.get("/metrics")
async def metrics():
    models = []
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as f:
            for row in csv.DictReader(f):
                name = row["Model"].strip()
                models.append({
                    "model": name,
                    "mae": round(float(row["MAE"]), 2),
                    "rmse": round(float(row["RMSE"]), 2),
                    "wape": round(float(row["WAPE"]), 2),
                    "is_baseline": "baseline" in name.lower(),
                    "is_deployed": name.lower() == "random forest",
                })

    # Aggregate RF feature importances back to the 10 input features
    imps = rf_estimator.feature_importances_
    n_num = len(NUM_FEATURES)
    grouped = {feat: round(float(imps[i]) * 100, 2) for i, feat in enumerate(NUM_FEATURES)}
    offset = n_num
    for col, cats in zip(CAT_FEATURES, ohe.categories_):
        k = len(cats)
        grouped[col] = round(float(imps[offset : offset + k].sum()) * 100, 2)
        offset += k

    feature_importances = sorted(
        [{"feature": FEATURE_LABELS[k], "importance_pct": v} for k, v in grouped.items()],
        key=lambda x: x["importance_pct"],
        reverse=True,
    )

    return {
        "model_version": MODEL_VERSION,
        "models": models,
        "feature_importances": feature_importances,
        "cold_start_info": {
            "known_movies_count": len(TRAINED_FILMS),
            "known_screens_count": len(TRAINED_CINEMAS),
            "known_showtimes_count": len(TRAINED_SHOWTIMES),
        },
    }


@app.post("/forecast")
async def forecast(req: ForecastRequest):
    try:
        start = date.fromisoformat(req.forecast_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc

    dates = [start + timedelta(days=i) for i in range(req.days)]
    results = []
    for dt in dates:
        single = PredictRequest(
            film_code=req.film_code,
            cinema_code=req.cinema_code,
            show_time=req.show_time,
            capacity=req.capacity,
            ticket_price=req.ticket_price,
            forecast_date=dt.isoformat(),
        )
        pred = run_single_prediction(single, dt, include_contributions=False)
        results.append({
            "date": pred["forecast_date"],
            "day": pred["day"],
            "is_weekend": dt.weekday() >= 5,
            "tickets": pred["predicted_tickets"],
            "interval_low": pred["interval_low"],
            "interval_high": pred["interval_high"],
            "baseline": pred["baseline_prediction"],
            "occ_pct": pred["predicted_occupancy"],
            "revenue": round(pred["predicted_tickets"] * req.ticket_price, 2),
            "food": pred["food"],
            "schedule": pred["schedule"],
            "tip": marketing_tip(pred["predicted_occupancy"]),
        })

    total_tickets = sum(r["tickets"] for r in results)
    total_revenue = round(sum(r["revenue"] for r in results), 2)
    avg_occ = round(sum(r["occ_pct"] for r in results) / len(results), 1)
    peak = max(results, key=lambda r: r["tickets"])
    total_food = {
        key: sum(r["food"][key] for r in results)
        for key in ("popcorn", "drinks", "snacks", "combos")
    }

    return {
        "forecast": results,
        "total_tickets": total_tickets,
        "total_revenue": total_revenue,
        "avg_occ": avg_occ,
        "peak": peak,
        "total_food": total_food,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
