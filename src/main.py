"""FastAPI application: demo backend services behind the secure gateway.

The endpoints themselves contain no security logic at all - authentication,
authorization, rate limiting, filtering and risk scoring all happen in the
gateway middleware (src/gateway.py). Run with:

    uvicorn src.main:app --reload
"""
from fastapi import FastAPI

from src import gateway

app = FastAPI(
    title="Intelligent Secure API Gateway",
    description=(
        "Master thesis prototype: secure API integration supported by "
        "intelligent mechanisms (hybrid rule-based + ML risk engine)."
    ),
    version="1.0.0",
)

# Register the security pipeline for every request.
app.middleware("http")(gateway.gateway_middleware)


# ---------------------------------------------------------------------------
# Demo backend endpoints
# ---------------------------------------------------------------------------
@app.get("/public")
async def public():
    """Open endpoint: no token required (still rate limited + risk scored)."""
    return {"message": "Public information", "service": "status", "healthy": True}


@app.get("/users")
async def users():
    """Requires a valid user or admin token."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "role": "user"},
            {"id": 2, "name": "Bob", "role": "user"},
        ]
    }


@app.get("/orders")
async def orders():
    """Requires a valid user or admin token."""
    return {
        "orders": [
            {"id": 101, "item": "Laptop", "status": "shipped"},
            {"id": 102, "item": "Phone", "status": "processing"},
        ]
    }


@app.post("/payments")
async def payments():
    """Sensitive endpoint: valid authentication is mandatory."""
    return {"status": "payment_accepted", "transaction_id": "txn-demo-001"}


@app.get("/admin")
async def admin():
    """Restricted endpoint: admin role only."""
    return {"message": "Admin panel", "settings": {"maintenance_mode": False}}
