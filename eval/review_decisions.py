"""Review decisions for eval/candidates.jsonl, made by an AI assistant (Claude) at the
project owner's request, and recorded here so they can be audited and changed.

    uv run python -m eval.review_decisions

Rules applied:
- keep: realistic, self-contained question; answer correct and fully supported by the quote
- edit: good question that needed context (which scheme/product) or a corrected answer
- reject: trivia (form fields, circular numbers, phone numbers), vague ("this context"),
  answers that only point elsewhere, evidence too short to match reliably, or two-passage
  questions that glue unrelated facts together
Also adds hand-written multi-part questions, whose quotes are checked against the documents.
"""

import sys

from documind.chunking.chunkers import load_parsed
from documind.config import get_settings
from eval.testset import CANDIDATES_PATH, Evidence, Record, load_records, locate_quote, save_records

REVIEWER = "Claude (AI assistant)"

KEEP = """h001 s001 s003 s004 s006 s010 s011 s016 s018 s020 s022 s023 s027 s031 s034 s036 s037
s042 s044 s045 s046 s047 s048 s049 s051 s053 s056 s057 s058 s059 s062 s063 s064 s065 s068
s069 s070 s071 s072 s073 s075 s079 s080 s081 s082 s084 s085 s087 s088 s089 s091 s092 s094
s096 s098 s099 s100 s102 s103 s104 s105 s107 s109 s110 s111 s113 s115 s116 s118 s120 s121
s122 s123""".split()

KEEP_UNANSWERABLE = True  # all 26 are clearly outside the corpus or rest on a false premise

# id -> (question, answer); None keeps the original text
EDITS = {
    "s002": (
        "How much can a resident individual lend to a non-resident relative under LRS in "
        "a financial year?",
        "The loan must stay within the overall LRS limit of USD 2,50,000 per financial year.",
    ),
    "s007": (
        "Under FEMA, from how many residential properties in India can a non-resident "
        "repatriate the sale proceeds?",
        "From not more than two residential properties.",
    ),
    "s009": (
        "Under RBI's foreign exchange rules on deposits and accounts, for how long can an "
        "escrow account be kept open at most?",
        "No more than seven years; no operations are allowed after seven years from the "
        "date the account was opened.",
    ),
    "s012": (
        "Under the Money Transfer Service Scheme (MTSS), how often must Indian Agents "
        "carry out spot audits of their Sub Agents?",
        "At least once a month, covering all Sub Agent locations.",
    ),
    "s013": (
        "Under the Money Transfer Service Scheme (MTSS), how often must Indian Agents "
        "carry out due diligence checks on their Sub Agents?",
        "On a regular basis, at least once a year.",
    ),
    "s017": (
        "Under RBI's foreign investment rules, what is the minimum price at which "
        "convertible instruments can be converted into shares?",
        "The conversion price can never be lower than the fair value worked out at the "
        "time the instruments were issued.",
    ),
    "s021": (
        "When a non-resident provides credit enhancement for a rupee debt instrument, "
        "what minimum average maturity must the underlying debt have?",
        "A minimum average maturity of three years.",
    ),
    "s026": (
        "In the Bharat Bill Payment System, what is an Agent Institution?",
        "An entity certified by NBBL that provides the physical or digital customer "
        "interface for bill payments.",
    ),
    "s028": (
        "In the Bharat Bill Payment System, what credits and debits are allowed in the "
        "escrow account of a Customer Operating Unit (COU)?",
        "Credit of funds collected from customers; debits for settlement of BBPS "
        "transactions; credits or debits for failed or disputed transactions; and "
        "recovery of charges or commissions on bill payments.",
    ),
    "s030": (
        "For an overseas investment in an IFSC, how long does the financial services "
        "regulator have to decide on the approval?",
        "45 days from receipt of a complete application; otherwise it is deemed approved.",
    ),
    "s035": (
        "Under RBI's Payment Aggregator directions, on what principle must access to "
        "applications be granted?",
        "On the principle of least privilege and 'need to know', in line with job "
        "responsibilities.",
    ),
    "s039": (
        "By when must payment system providers comply with RBI's directions on "
        "authentication mechanisms for digital payments?",
        "By April 1, 2026, unless a specific provision says otherwise.",
    ),
    "s052": (
        "What must a borrower do if a person on the list of wilful defaulters is on its "
        "board or in charge of its management?",
        "Take expeditious and effective steps to remove that person from the board or "
        "from being in charge of its management.",
    ),
    "s066": (
        "When a bank prices a loan, can the business strategy component of the spread be negative?",
        "No. The spread components (business strategy and credit risk premium) must be "
        "positive or zero.",
    ),
    "s067": (
        "For residential mortgage-backed securitisations, what is the minimum reserve "
        "floor when releasing excess credit enhancement?",
        "20 per cent of the initial credit enhancement.",
    ),
    "s078": (
        "Once a complaint about credit information is resolved, within how many working "
        "days must compensation be credited to the complainant's bank account?",
        "Within five working days of the complaint being resolved.",
    ),
    "s083": (
        "Is there a fixed maximum quantity of gold a bank can lend to one borrower under "
        "the Gold Monetization (GML) scheme?",
        "The directions don't fix a number: each bank must set a per-borrower limit in "
        "its board-approved lending and risk management policy for GML.",
    ),
    "s086": (
        "If an FCNR(B) deposit is renewed within 14 days after it matures, what interest "
        "rate applies to the renewed deposit?",
        "The rate for the renewal period prevailing on the maturity date or on the date "
        "renewal is sought, whichever is lower.",
    ),
    "s097": (
        "What must a bank submit with its application to offer a transactional digital "
        "banking facility?",
        None,
    ),
    "s106": (
        "For card e-mandates on recurring payments, up to what amount per transaction can "
        "payments go through without an additional factor of authentication (AFA)?",
        "Up to ₹15,000 per transaction.",
    ),
    "s119": (
        "How often must a bank run vulnerability assessments on its digital payment "
        "applications and infrastructure?",
        "At least every six months (penetration tests at least once a year).",
    ),
    "m001": (
        "If I have a minimum-details PPI wallet, by when must it be converted to full KYC, "
        "and can I link it to UPI before that?",
        "It must be converted to full-KYC within 24 months of issue, otherwise no further "
        "credit is allowed; only full-KYC PPIs can be linked to UPI for payments.",
    ),
    "m010": (
        "In the Bharat Bill Payment System, who handles clearing and settlement, and where "
        "must a non-bank operating unit keep the funds?",
        "NBBL, as the Bharat Bill Pay Central Unit, does clearing and settlement; a non-bank "
        "BBPOU must open an escrow account with a scheduled commercial bank used only for "
        "BBPS transactions.",
    ),
    "m015": (
        "By when must payment system providers comply with RBI's 2025 authentication "
        "directions, and what later deadline applies to card issuers for cross-border "
        "card-not-present transactions?",
        "General compliance is due by April 1, 2026; card issuers must put in place a "
        "risk-based mechanism for cross-border card-not-present transactions by "
        "October 1, 2026.",
    ),
}

REJECT_REASONS = {
    "s005": "form certificate text, not a realistic question",
    "s008": "vague ('in this context')",
    "s014": "administrative detail nobody would ask",
    "s015": "circular date trivia",
    "s019": "answer only restates another rule",
    "s024": "form undertaking text",
    "s025": "abbreviation trivia",
    "s029": "evidence includes an amendment footnote; answer is a long list",
    "s032": "phone number trivia",
    "s033": "circular number trivia",
    "s038": "circular number trivia",
    "s040": "circular number trivia",
    "s041": "vague, no specific fact",
    "s043": "answer restates the question",
    "s050": "obscure (UAPA nodal officer)",
    "s054": "vague ('these directions')",
    "s055": "answer has no content",
    "s060": "table row, not a question people ask",
    "s061": "technical calculation step without context",
    "s074": "vague answer",
    "s076": "reporting-format field trivia",
    "s077": "reporting-format field trivia",
    "s090": "reporting-template row",
    "s093": "convoluted; answer garbles the rule",
    "s095": "legal-basis trivia",
    "s101": "vague ('the Directions')",
    "s108": "circular number trivia",
    "s112": "worked-example figure, not a rule",
    "s114": "evidence too short to match reliably",
    "s117": "answer only points to another circular",
}
MULTI_KEEP = {"m001", "m010", "m015"}
MULTI_REJECT_REASON = "glues two unrelated facts together; not a question a person would ask"

# Hand-written multi-part questions: (question, answer, [(doc_id, quote), ...])
HANDWRITTEN_MULTI = [
    (
        "How much money can I send abroad in a year under LRS, and can I use it to buy a "
        "property abroad?",
        "Up to USD 2,50,000 per financial year; buying immovable property abroad is one of the "
        "permitted capital account transactions under LRS.",
        [
            (
                "rbi-10192",
                "The limit of USD 2,50,000 per Financial Year (FY) under the Scheme "
                "also includes/subsumes remittances for current account transactions",
            ),
            (
                "rbi-10192",
                "acquisition of immovable property abroad, Overseas Direct Investment "
                "(ODI) and Overseas Portfolio Investment (OPI)",
            ),
        ],
    ),
    (
        "A bank activated a credit card I never asked for and billed me. What must it do, and "
        "am I responsible if that card is misused?",
        "It must reverse the charges immediately and pay you a penalty of twice the reversed "
        "charges; any loss from misuse of an unsolicited card is the card-issuer's "
        "responsibility, not yours.",
        [
            (
                "rbi-13155",
                "the card-issuer shall not only reverse the charges forthwith, but also "
                "pay a penalty without demur to the recipient amounting to twice the value of the "
                "charges reversed",
            ),
            (
                "rbi-13155",
                "any loss arising out of misuse of such unsolicited cards shall be the "
                "responsibility of the card-issuer only",
            ),
        ],
    ),
    (
        "If my savings account falls below the minimum balance, how long does the bank give me "
        "before charging, and how must the charge be calculated?",
        "The bank must notify you, and penal charges apply only if the balance isn't restored "
        "within a month of the notice; the charge must be directly proportionate to the "
        "shortfall.",
        [
            (
                "rbi-13140",
                "in the event of the minimum balance not being restored in the account "
                "within a month from the date of notice, penal charges shall be applicable",
            ),
            (
                "rbi-13140",
                "The penal charges shall be directly proportionate to the extent of "
                "shortfall observed.",
            ),
        ],
    ),
    (
        "If an unauthorised electronic transaction happens on my account because of a breach "
        "elsewhere in the system, what is my liability if I report it within three working "
        "days, and what if I report it within five?",
        "Within three working days: zero liability. Within four to seven working days: "
        "liability is limited to the transaction value or the cap in Table 1 (for example "
        "₹5,000 for BSBD accounts), whichever is lower.",
        [
            (
                "rbi-13140",
                "the customer notifies the bank within three working days of "
                "receiving the communication from the bank regarding the unauthorised transaction",
            ),
            (
                "rbi-13140",
                "within four to seven working days of receiving a communication of the "
                "transaction, the per transaction liability of the customer shall be limited to "
                "the transaction value or the amount mentioned in Table 1, whichever is lower",
            ),
        ],
    ),
    (
        "If I dispute a credit card transaction as fraud, can the bank charge me while it "
        "investigates, and how quickly must I report an unauthorised transaction to have zero "
        "liability when the fault lies elsewhere in the system?",
        "No charges can be levied on transactions disputed as fraud until the dispute is "
        "resolved; reporting within three working days of the bank's communication gives zero "
        "liability.",
        [
            (
                "rbi-13155",
                "No charges shall be levied on transactions disputed as ‘fraud’ by the "
                "cardholder until the dispute is resolved.",
            ),
            (
                "rbi-13140",
                "the customer notifies the bank within three working days of "
                "receiving the communication from the bank regarding the unauthorised transaction",
            ),
        ],
    ),
]


def find_evidence(docs: dict, doc_id: str, quote: str) -> Evidence:
    for p in docs[doc_id]["paragraphs"]:
        if exact := locate_quote(quote, p["text"]):
            return Evidence(doc_id, exact, p["para"], p["page_start"])
    raise ValueError(f"quote not found in {doc_id}: {quote[:60]}")


def main() -> int:
    records = load_records(CANDIDATES_PATH)
    records = [r for r in records if not r.id.startswith("x")]  # re-runnable
    by_id = {r.id: r for r in records}
    unknown = (set(KEEP) | set(EDITS) | set(REJECT_REASONS)) - set(by_id)
    if unknown:
        print("Unknown ids:", sorted(unknown))
        return 1

    for r in records:
        note = r.note.split(" | REVIEW")[0]
        if r.id in EDITS:
            question, answer = EDITS[r.id]
            r.question = question or r.question
            r.answer = answer or r.answer
            r.status, reason = "edited", "added context or corrected the answer"
        elif r.id in KEEP or (r.kind == "unanswerable" and KEEP_UNANSWERABLE):
            r.status, reason = "approved", "kept"
        elif r.kind == "multi" and r.id not in MULTI_KEEP:
            r.status, reason = "rejected", MULTI_REJECT_REASON
        elif r.id in REJECT_REASONS:
            r.status, reason = "rejected", REJECT_REASONS[r.id]
        else:
            print(f"No decision for {r.id}")
            return 1
        r.note = f"{note} | REVIEW ({REVIEWER}): {r.status}: {reason}".lstrip(" |")

    docs = {d["doc_id"]: d for d in load_parsed(get_settings().data_dir / "processed")}
    for i, (question, answer, quotes) in enumerate(HANDWRITTEN_MULTI, 1):
        records.append(
            Record(
                id=f"x{i:03d}",
                question=question,
                answer=answer,
                kind="multi",
                qtype="multi_part",
                evidence=[find_evidence(docs, d, q) for d, q in quotes],
                origin="handwritten",
                status="approved",
                note=f"REVIEW ({REVIEWER}): written by the reviewer; quotes verified",
            )
        )

    save_records(records, CANDIDATES_PATH)
    kept = [r for r in records if r.status in ("approved", "edited")]
    for kind in ("single", "multi", "unanswerable"):
        group = [r for r in records if r.kind == kind]
        print(f"{kind:<13} kept {sum(r in kept for r in group):>3} of {len(group)}")
    print(f"total kept: {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
