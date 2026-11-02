# Instagram Screenshot → Telegram → Supabase

A small Python Telegram bot. No AI, no separate API server, and no image storage.
It uses local Tesseract OCR to read text visible in screenshots and saves records directly to Supabase.

## 1. Supabase setup
1. Open your Supabase project → SQL Editor → New query.
2. Paste all of `supabase/schema.sql`.
3. Run it.

## 2. Install Tesseract OCR (Windows)
Install the Tesseract OCR Windows installer, then ensure `tesseract.exe` is on your PATH.
A common install location is `C:\Program Files\Tesseract-OCR`.
Close and reopen the terminal after installation.
Check with:
```powershell
tesseract --version
```

## 3. Configure
Open `.env` and add your NEW rotated credentials:
- `SUPABASE_URL`
- `SUPABASE_KEY` (Supabase secret/service-role key; backend only)
- `TELEGRAM_BOT_TOKEN`
- `AUTHORIZED_USER_ID`

The previously shared credentials should be rotated before use.

## 4. Run on Windows PowerShell
```powershell
cd path\to\instagram_telegram_collector
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -m bot.main
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Add your team
1. Open the bot and send `/start` from the admin Telegram account.
2. Ask each team member to message the bot once and share their numeric Telegram user ID with you.
3. In the bot, run `/allow TELEGRAM_ID Full Name` for each member.
4. They send `/start` again, then send profile screenshots.

## Commands
- `/start`
- `/today`
- `/stats` (admin sees all active team members)
- `/team` (admin only)
- `/allow TELEGRAM_ID Full Name` (admin only)
- `/help`

## Notes
- This version extracts the first visible email using OCR. OCR can miss or misread text; use clear screenshots and manually verify leads.
- Only screenshots with an extracted email are added as creator leads. Screenshots without an email are recorded as submissions with `email_not_found`.
- Duplicate email addresses are prevented atomically in PostgreSQL. The first member to add an email receives the unique-lead credit; later submissions are recorded as duplicates.
- Username uniqueness is also enforced by a database index. If a different email is later found for the same username, review/merge manually rather than overwriting a creator.
- Screenshots are held in memory and are not stored in Supabase or on disk by this code.
- Keep the bot process running. For later deployment, use an always-on worker/service; a sleeping serverless function will not reliably receive Telegram polling updates or run scheduled jobs.
- Daily target is currently a display target (50). This minimal build does not include scheduled reminders or automated campaign sending.
