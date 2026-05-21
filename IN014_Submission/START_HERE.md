# START HERE — IN014 IT Support Tickets

**Project:** IN014 — IT Support Ticket Analytics  
**Team:** Miquel Raurich, Bruno Mazzoli, Pablo Bagan, Hugo

## Run everything (~2 min)

```bash
pip install -r requirements.txt
bash run_all.sh
```

This refreshes charts, metrics, and models from the **cleaned CSVs already in `data/cleaned/`**.  
To rebuild from scratch, add the course files to `data/raw/` (`it_support_tickets.csv`, `customer_info.csv`) and run again.

## Report (submit this)

- **`report.pdf`** — formatted for submission  
- **`report.md`** — same content, easy to edit  

Open **`outputs/charts/dashboard.html`** in a browser for the interactive view.

## Before you hand in

Feel free to tweak the personal reflections in **§6** of `report.md` (then re-export PDF if needed: `pandoc report.md -o report.pdf --toc`).

More detail: `README.md`. Ideas for later: `FUTURE_WORK.md`.
