"""
Generates sample_docs/lumora_employee_handbook.pdf -- a multi-page sample
document for trying the chatbot's document Q&A (and page citations).

Lumora Labs is a fictional company; every policy below is made up for
demonstration purposes.

    python scripts/make_sample_pdf.py
"""
import os

import pymupdf as fitz  # PyMuPDF

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "sample_docs", "lumora_employee_handbook.pdf")

SECTIONS = [
    ("1. Welcome to Lumora Labs", """
<p>Lumora Labs builds analytics software for renewable-energy operators. The
company was founded in 2014, is headquartered in Pune, and has offices in
Bengaluru and Singapore. As of this edition, Lumora employs about 1,200
people across engineering, data science, sales and operations.</p>
<p>This handbook explains the policies that apply to all full-time employees.
Where a local law grants more generous terms than this handbook, the local
law applies. Questions about any policy should go to the People Team at
people@lumoralabs.example.</p>
<p>The handbook is reviewed every year in January. Changes take effect on
1 February unless a different date is stated.</p>
"""),
    ("2. Working Hours and Hybrid Work", """
<p>Lumora uses flexible hours. Employees may start work any time between
8:00 and 11:00, provided they are available during <b>core hours of 10:30 to
16:00</b> (local office time), Monday to Friday. A standard working week is
40 hours.</p>
<p>Lumora follows a hybrid model: employees work from the office <b>three days
a week, on Tuesday, Wednesday and Thursday</b>. Monday and Friday are remote
days by default. Teams may agree on additional office days for workshops or
launches.</p>
<p>Employees who need a fully remote arrangement for more than four weeks
must get written approval from their manager and the People Team.</p>
"""),
    ("3. Leave Policy", """
<p><b>Annual leave.</b> Full-time employees receive <b>24 days of paid annual
leave</b> per calendar year, accruing at 2 days per month. Up to <b>10 unused
days may be carried forward</b> into the next year; any balance above 10 days
lapses on 31 March.</p>
<p><b>Sick leave.</b> Employees receive 12 days of paid sick leave per year.
A medical certificate is required for absences of <b>more than three
consecutive days</b>. Sick leave does not carry forward.</p>
<p><b>Parental leave.</b> The primary caregiver receives 26 weeks of paid
parental leave; the secondary caregiver receives 6 weeks. Parental leave can
be taken any time within 12 months of the birth or adoption.</p>
<p><b>Other leave.</b> Bereavement leave is 5 paid days. Lumora observes 11
public holidays per year; the list is published each December. Leave
requests are submitted in the HR portal, called <i>Compass</i>, at least two
weeks in advance for absences longer than three days.</p>
"""),
    ("4. Compensation and Payroll", """
<p>Salaries are paid on the <b>last working day of each month</b>. Payslips
are available in Compass. Annual salary reviews take place in April and any
increase applies from 1 April.</p>
<p>Employees are eligible for an annual <b>performance bonus of up to 15% of
base salary</b>, paid in April, based on company results and the employee's
performance rating. Employees who joined after 1 October are eligible for a
pro-rated bonus.</p>
<p>Lumora contributes to the statutory provident fund as required by law and
offers an optional employee stock purchase plan with a 10% discount.</p>
"""),
    ("5. Travel and Expenses", """
<p>All business travel must be approved by the employee's manager before
booking. Expense claims must be submitted in the <i>Ledger</i> expense portal
<b>within 30 days</b> of the expense, with itemised receipts. Claims
submitted later than 60 days are not reimbursed.</p>
<p><b>Flights.</b> Domestic flights are booked in economy class.
International flights longer than 6 hours may be booked in premium economy.</p>
<p><b>Hotels.</b> The nightly hotel cap is INR 7,000 for domestic travel and
USD 180 for international travel.</p>
<p><b>Meals.</b> The daily meal allowance is <b>INR 1,500 for domestic
travel</b> and <b>USD 60 for international travel</b>. Alcohol is not
reimbursed.</p>
"""),
    ("6. Learning and Development", """
<p>Every employee has an annual <b>learning budget of INR 50,000</b> for
courses, conferences and books. Unused budget does not roll over.</p>
<p>Professional certification fees are reimbursed in full once the employee
passes the exam. Employees also receive <b>5 paid learning days</b> per year
for study or conference attendance.</p>
<p>New managers complete the eight-week Lumora Leadership Programme within
their first six months in the role.</p>
"""),
    ("7. IT, Security and Devices", """
<p>Passwords must be at least <b>14 characters</b> long. Multi-factor
authentication (MFA) is mandatory for every company system. Passwords are
changed only when there is a suspected compromise, not on a fixed schedule.</p>
<p>Security incidents, including phishing emails that were clicked, must be
reported to security@lumoralabs.example <b>within 1 hour</b> of discovery.
A lost or stolen laptop must be reported <b>within 2 hours</b> so it can be
wiped remotely.</p>
<p>Laptops are refreshed every <b>3 years</b>. New employees also receive a
one-time <b>home-office allowance of INR 25,000</b> for a chair, desk or
monitor.</p>
"""),
    ("8. Code of Conduct", """
<p>Employees must act honestly, treat colleagues with respect and avoid
conflicts of interest. Any outside employment or board role must be
disclosed to the People Team before it starts.</p>
<p>Gifts or hospitality from customers or suppliers worth more than
<b>INR 3,000</b> must be declared in Compass within 7 days. Cash gifts must
never be accepted.</p>
<p>Concerns about misconduct can be raised anonymously through the ethics
hotline, which is run by an independent provider and available 24 hours a
day.</p>
"""),
    ("9. Performance Reviews and Probation", """
<p>New employees serve a <b>probation period of six months</b>. Performance
reviews take place <b>twice a year, in April and October</b>, using a rating
scale from 1 (does not meet expectations) to 5 (exceptional).</p>
<p>Employees rated 2 or below receive a written 90-day improvement plan with
monthly check-ins. Promotions are decided in the April review cycle.</p>
"""),
    ("10. Health and Wellbeing", """
<p>Lumora provides family-floater health insurance with cover of
<b>INR 10 lakh</b>, covering the employee, spouse or partner, children and
two parents. An annual health check-up is free for employees.</p>
<p>The Employee Assistance Programme offers confidential counselling 24 hours
a day, seven days a week, by phone or video. Employees may also claim up to
INR 12,000 per year towards gym or fitness memberships.</p>
"""),
    ("11. Leaving Lumora", """
<p>The notice period is <b>60 days</b> after probation and <b>30 days</b>
during probation. Unused annual leave is paid out at the end of employment.</p>
<p>The full and final settlement is completed <b>within 45 days</b> of the
last working day. Company devices must be returned on or before the last
working day.</p>
"""),
]

CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 22pt; color: #1f3a5f; margin-bottom: 4pt; }
h2 { font-size: 15pt; color: #1f3a5f; margin-top: 14pt; }
p { margin-bottom: 8pt; text-align: justify; }
.sub { color: #555; font-size: 12pt; }
"""


def build_html() -> str:
    parts = ["<h1>Lumora Labs Employee Handbook</h1>",
             "<p class='sub'>Policies for full-time employees. Fictional company, "
             "created as a sample document.</p>"]
    for title, body in SECTIONS:
        parts.append(f"<h2>{title}</h2>{body}")
    return "".join(parts)


def main():
    story = fitz.Story(html=build_html(), user_css=CSS)
    writer = fitz.DocumentWriter(OUT)
    mediabox = fitz.paper_rect("a5")  # small pages -> a properly multi-page sample
    where = mediabox + (40, 40, -40, -50)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Footer with page numbers, so citations can be checked against the PDF.
    doc = fitz.open(OUT)
    for i, page in enumerate(doc, start=1):
        rect = page.rect
        page.insert_text((rect.width / 2 - 30, rect.height - 22),
                         f"Page {i} of {doc.page_count}", fontsize=8, color=(0.4, 0.4, 0.4))
    doc.saveIncr()
    print(f"Wrote {OUT} ({doc.page_count} pages)")
    doc.close()


if __name__ == "__main__":
    main()
