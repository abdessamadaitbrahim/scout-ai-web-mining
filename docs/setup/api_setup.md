# API Setup Guide

## Football-Data.org API

1. Go to https://www.football-data.org/client/register
2. Register for a free account (no credit card required)
3. Your API key will be emailed to you
4. Free tier: **10 requests/minute**, covers Premier League, La Liga, Bundesliga, Serie A, Ligue 1
5. Add to your `.env` file:
   ```
   FOOTBALL_DATA_API_KEY=your_key_here
   ```

### Endpoints Used
| Endpoint | Description |
|----------|-------------|
| `/competitions/{id}/standings` | League standings |
| `/competitions/{id}/scorers` | Top goalscorers |
| `/competitions/{id}/matches` | Recent matches |

### Rate Limiting
The collector in `collectors/axe1_content_mining.py` automatically respects the
rate limit via `polite_delay()` in `utils/helpers.py` (1.5–3.0 seconds between requests).

---

## NewsAPI

1. Go to https://newsapi.org/register
2. Register for a free developer account
3. Your API key will be shown immediately after registration
4. Free tier: **100 requests/day**, limited to articles from the last 30 days
5. Add to your `.env` file:
   ```
   NEWS_API_KEY=your_key_here
   ```

### Endpoints Used
| Endpoint | Description |
|----------|-------------|
| `/v2/everything` | Full-text article search across all sources |

### Queries Run
For each of the 10 target players, two queries are run in French and English:
- `"{player_name} transfert"` / `"{player_name} transfer"`
- `"{player_name} {club_name}"`

---

## Running Without API Keys (Demo Mode)

If you don't have API keys, use demo mode which uses cached data from `data/`:

```bash
python main.py --demo
python run_all.py --demo
```

Demo mode skips live API calls and uses:
- Pre-collected JSON files in `data/` (included in the repository)
- Simulated web logs (generated internally by `axe3_usage_mining.py`)
- Simulated Google Trends data (generated internally)

> ⚠️ **Security Reminder**: If you received this project with API keys hardcoded
> in the source code, those keys are compromised. Rotate them immediately using
> the links above.
