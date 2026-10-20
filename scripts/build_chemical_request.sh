#!/usr/bin/env bash
set -euo pipefail
cd /home/siddhesh/Projects/saafscan

F=CHEMICAL_REQUEST.docx

officecli create "$F" --force

# --- Page setup: A4, 2.2 cm margins -------------------------------------
officecli set "$F" /body/sectPr[1] \
  --prop pageWidth=11906 --prop pageHeight=16838 \
  --prop marginTop=1247 --prop marginBottom=1247 \
  --prop marginLeft=1247 --prop marginRight=1247 >/dev/null

# --- Base style ---------------------------------------------------------

# --- Date ---------------------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="20 October 2026" \
  --prop align=right --prop spaceAfter=240 >/dev/null

# --- Recipient block ----------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="To," --prop spaceAfter=0 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="The Head of the Department," --prop spaceAfter=0 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="[Department Name]" --prop spaceAfter=0 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="[Institute Name]" --prop spaceAfter=0 >/dev/null

# --- Subject (p[6]) -----------------------------------------------------
officecli add "$F" "/body" --type paragraph \
  --prop text="Subject: Reagents and consumables for the Tech4Food 2026 final round (Team IRIS)" \
  --prop b=1 --prop color=1F4E79 --prop size=12 \
  --prop spaceBefore=280 --prop spaceAfter=200 >/dev/null
officecli set "$F" "/body/p[6]" --prop pbdr.bottom="single;6;1F4E79;6" >/dev/null

# --- Salutation ---------------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="Respected Sir/Madam," --prop spaceAfter=160 >/dev/null

# --- Body ---------------------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="Team IRIS has qualified for the final round of Tech4Food 2026, to be held at PCCOE on 24 October 2026. Our project detects milk adulteration from a single photograph of a reagent strip. Iodine identifies starch; methylene blue identifies detergent." --prop spaceAfter=170 >/dev/null

officecli add "$F" "/body" --type paragraph --prop text="We trained our model on 534 synthetic strip images. The final will run on real reagent strips, so we need to photograph prepared strips in at least three lighting conditions and retrain before 23 October. That is what the chemicals are for." --prop spaceAfter=170 >/dev/null

officecli add "$F" "/body" --type paragraph --prop text="We request the following, in small quantities:" --prop spaceAfter=100 >/dev/null

# --- Table --------------------------------------------------------------
DATA='"Item","Quantity","Purpose"
"Iodine solution (Lugol’s or tincture, dilute)","10 mL","Starch test"
"Methylene blue (dilute aqueous solution)","10 mL","Detergent test"
"Distilled water","50 mL","Dilution"
"Filter paper, Whatman No. 1, 11 cm circles","2–3","Strip stock, about 20 strips per circle"
"Disposable Pasteur pipettes","10","Measured drop application"
"Amber dropper bottle, 20–30 mL","1","Storage and transport"
"Nitrile gloves","4–6 pairs","Both stains persist on skin"
"SDS (sodium dodecyl sulphate), optional","a few mL","More reproducible than dish soap"'

ROWS=$(printf '%s' "$DATA" | tr '\n' ';' | sed 's/;$//')

officecli add "$F" "/body" --type table --prop cols=3 \
  --prop layout=fixed --prop width=9412 \
  --prop colWidths=4800,1400,3212 \
  --prop border.all="single;4;C9D3DE;0" \
  --prop data="$ROWS" >/dev/null

# header row: dark blue fill, white bold text, vertically centred
for c in 1 2 3; do
  officecli set "$F" "/body/table[1]/tr[1]/tc[${c}]" \
    --prop fill=1F4E79 --prop bold=1 --prop color=FFFFFF --prop valign=center >/dev/null
done
# all cells vertically centred
for i in 2 3 4 5 6 7 8 9; do
  for c in 1 2 3; do
    officecli set "$F" "/body/table[1]/tr[${i}]/tc[${c}]" --prop valign=center >/dev/null
  done
done
# subtle banding on alternate data rows
for i in 3 5 7; do
  for c in 1 2 3; do
    officecli set "$F" "/body/table[1]/tr[${i}]/tc[${c}]" --prop fill=F4F8FC >/dev/null
  done
done
# optional row: quiet grey italic so it reads as a bonus, not a demand
for c in 1 2 3; do
  officecli set "$F" "/body/table[1]/tr[9]/tc[${c}]" --prop fill=F2F2F2 --prop italic=1 --prop color=595959 >/dev/null
done

# --- After the table ----------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="One test uses one or two drops, so 10 mL of each reagent covers more than a hundred tests, well beyond what the preparation period and the final require. Both solutions can be dilute. If you have a concentration you prefer for a starch demonstration, we will use what you recommend." --prop spaceBefore=200 --prop spaceAfter=170 >/dev/null

officecli add "$F" "/body" --type paragraph --prop text="Two points we need your guidance on:" --prop spaceAfter=80 >/dev/null

officecli add "$F" "/body" --type paragraph --prop listStyle=ordered --prop text="Whether these quantities may be carried to PCCOE on 24 October, or should be arranged locally on the day." >/dev/null
officecli add "$F" "/body" --type paragraph --prop listStyle=ordered --prop text="Whether a lab bench is available on 20–22 October for preparing and photographing the strips." >/dev/null

officecli add "$F" "/body" --type paragraph --prop text="We will work with gloves, keep every bottle sealed and labelled, use dilute solutions only, and return whatever is left. Nothing leaves the lab without your approval." --prop spaceBefore=200 --prop spaceAfter=170 >/dev/null

# --- Close --------------------------------------------------------------
officecli add "$F" "/body" --type paragraph --prop text="Thanking you," --prop spaceBefore=200 --prop spaceAfter=160 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="Yours faithfully," --prop spaceAfter=320 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="[Name]" --prop b=1 --prop spaceAfter=0 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="Team IRIS" --prop spaceAfter=0 >/dev/null
officecli add "$F" "/body" --type paragraph --prop text="[Roll numbers] · [Department]" >/dev/null

# --- Reminder comment on the first fill-in field ------------------------
# Sits in Word's margin, never prints, so the letter itself stays clean.
officecli add "$F" "/body/p[4]" --type comment \
  --prop author="Team IRIS" --prop initials="IRIS" --prop range=true \
  --prop text="Fill in the four bracketed fields (department, institute, name, roll numbers), then delete this comment before printing or sending." >/dev/null

echo "BUILT"
