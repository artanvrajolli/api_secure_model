# Synthetic API Traffic Dataset

`api_traffic_dataset.csv` is a **fully synthetic** dataset produced by
`src/generate_dataset.py` (random seed 42, deterministic). It simulates one
day of API gateway traffic and stands in for the production traffic logs a
real deployment would collect. No real user data is contained.

Regenerate with:

```bash
python -m src.generate_dataset
```

## Composition

- **12,000 records total**
- **10,200 normal requests (85%)** from ~40 regular clients
- **1,800 anomalous requests (15%)** from ~15 attacker identities,
  evenly split across seven anomaly types

| `anomaly_type` | Simulated behaviour | Main signals |
|---|---|---|
| `brute_force` | Repeated failed authentications, incl. "low and slow" | high `failed_auth_count`, `token_valid=0`, status 401 |
| `rate_abuse` | Excessive request volume / rate-limit abuse | very high `requests_per_minute`, status 429 |
| `unauthorized_access` | Users probing `/admin` and `/payments` | status 403, high `endpoint_sensitivity` |
| `injection_payload` | SQLi / XSS / traversal payloads | `payload_suspicious=1`, status 400 |
| `slow_response` | Abnormal latency (resource exhaustion symptom) | `response_time_ms` 900–8000 |
| `oversized_request` | Abnormally large bodies | `request_size_bytes` 20 KB–500 KB |
| `scanning_4xx` | Endpoint scanning / probing | repeated 401/403/404, elevated rpm |

Normal traffic intentionally **overlaps** with mild anomalies (≈5%
legitimate activity bursts, ≈3% slow heavy queries, ≈2% large uploads,
occasional mistyped passwords) so the classification task is realistic.

## Columns

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO 8601 | Time of the request (simulated day 2026-06-01) |
| `client_id` | str | Client identity (`client_*` normal, `attacker_*` anomalous) |
| `ip_address` | str | Client IP (internal `10.0.*` vs TEST-NET `203.0.113.*`) |
| `endpoint` | str | `/public`, `/users`, `/orders`, `/payments`, `/admin` |
| `http_method` | str | GET / POST |
| `status_code` | int | HTTP response status |
| `response_time_ms` | float | Server response time in milliseconds ★ |
| `request_size_bytes` | int | Size of the request body/headers ★ |
| `requests_per_minute` | int | Client's request rate in the current window ★ |
| `failed_auth_count` | int | Client's recent failed authentications ★ |
| `token_valid` | 0/1 | 1 = valid token or none required, 0 = invalid ★ |
| `user_role` | str | `anonymous`, `user`, `admin` |
| `endpoint_sensitivity` | float | Business criticality 0.1–1.0 ★ |
| `payload_suspicious` | 0/1 | Attack signature found by the request filter ★ |
| `anomaly_label` | 0/1 | **Target:** 0 = normal, 1 = anomalous |
| `anomaly_type` | str | Anomaly category (`none` for normal traffic) |

★ = feature used by the ML model (`config.FEATURE_COLUMNS`). The label and
`anomaly_type` are never shown to the model at inference time;
`anomaly_type` exists for analysis and thesis discussion only.
