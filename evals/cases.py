"""Fixed evaluation cases.

Gold data describes what the documents say. Never edit a case to match a
model's output. If a case is wrong about the documents, fix it in its own
commit that quotes the source text.
"""

CASES = [
    {
        "id": "cake-shared",
        "question": (
            "When employees are eating cake in a shared space, "
            "who must be offered a slice?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "HR Policy|2.0|7. Shared Refrigerator Policy > 7.2 Cake-Sharing Default"
        ],
        "required_facts": [["leader of the other pod"], ["slice"]],
        "stale_facts": [],
    },
    {
        "id": "dog-adoption",
        "question": "How many paid days off does an employee get for adopting a dog?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": ["HR Policy|2.0|5. Pet Adoption Leave > 5.1 Leave Entitlement"],
        "required_facts": [
            ["7 days", "seven days", "7 paid days", "7 day", "seven day"],
            ["5 days", "five days"],
        ],
        "stale_facts": [],
    },
    {
        "id": "boss-error",
        "question": (
            "Under the HR Policy, how long must an employee wait "
            "before pointing out a manager's mistake?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": ["HR Policy|2.0|6. Boss Error Grace Period"],
        "required_facts": [["30 minutes"]],
        "stale_facts": [],
    },
    {
        "id": "dress-code",
        "question": "What clothing does the HR dress code prohibit?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": ["HR Policy|2.0|4. Dress Code > 4.1 Prohibited Attire"],
        "required_facts": [["suits"], ["ties"]],
        "stale_facts": [],
    },
    {
        "id": "weekend-fridge",
        "question": (
            "Under the HR Policy, what happens to food left in the "
            "shared refrigerator over a weekend?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "HR Policy|2.0|7. Shared Refrigerator Policy > "
            "7.3 Weekend Abandonment Consequence"
        ],
        "required_facts": [["abandoned"], ["spoonful"]],
        "stale_facts": [],
    },
    {
        "id": "caffeine",
        "question": "What is the recommended maximum caffeine per day?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Health & Wellness Policy|1.0|5. Caffeine Guidelines > 5.1 Daily Limit"
        ],
        "required_facts": [["400 mg"]],
        "stale_facts": [],
    },
    {
        "id": "gym",
        "question": (
            "How many gym sessions per week are employees expected to "
            "complete, and how long is each one?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Health & Wellness Policy|1.0|3. Gym Routine Requirements > "
            "3.1 Minimum Requirement"
        ],
        "required_facts": [["three"], ["45 minutes"]],
        "stale_facts": [],
    },
    {
        "id": "interns",
        "question": (
            "Under the Health Policy, are interns required to meet "
            "the gym-attendance minimum?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": ["Health & Wellness Policy|1.0|2. Scope"],
        "required_facts": [["interns"], ["exempt"]],
        "stale_facts": [],
    },
    {
        "id": "video-game-minutes",
        "question": "How many minutes of video games may an employee play per workday?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|2.0|3. Video Game Time > 3.1 Daily Allowance"
        ],
        "required_facts": [["45 minutes"]],
        "stale_facts": [],
    },
    {
        "id": "tokens-current",
        "question": (
            "How many tokens does each employee receive at the start "
            "of a six-hour cycle?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|2.0|6. Token Allocation > 6.1 Allocation Amount"
        ],
        "required_facts": [["500,000", "five hundred thousand"]],
        "stale_facts": ["1,000,000", "one million"],
    },
    {
        "id": "tokens-v1",
        "question": (
            "What did Time and Usage Policy 1.0 issue for tokens "
            "at the start of each cycle?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|1.0|5. Token Allocation > 5.1 Allocation Amount"
        ],
        "required_facts": [["1,000,000", "one million"]],
        "stale_facts": ["500,000", "five hundred thousand"],
    },
    {
        "id": "foosball-winner",
        "question": (
            "If you win a foosball match, what happens to the other player's tokens?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|2.0|4. Foosball Time and the "
            "Winner-Takes-Tokens Rule > 4.2 Winner-Takes-Tokens Rule"
        ],
        "required_facts": [["remaining token"], ["winner"]],
        "stale_facts": [],
    },
    {
        "id": "nuclear-shelter",
        "question": (
            "Where should employees shelter when a nuclear detonation is imminent?"
        ),
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Preparedness Policy|2.0|4. Nuclear Apocalypse Protocol — "
            "Updated > 4.1 Shelter Location"
        ],
        "required_facts": [["break room"], ["refrigerator"]],
        "stale_facts": ["under their desks"],
    },
    {
        "id": "hazmat",
        "question": "Who gets a hazmat suit in a nuclear emergency?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Preparedness Policy|2.0|4. Nuclear Apocalypse Protocol — "
            "Updated > 4.2 Hazmat Suit Eligibility"
        ],
        "required_facts": [["top 10"], ["foosball"]],
        "stale_facts": [],
    },
    {
        "id": "ai-apocalypse",
        "question": "What should employees do first in an AI apocalypse?",
        "route": "lookup",
        "expect_status": "answered",
        "gold_chunks": [
            "Preparedness Policy|2.0|7. AI Apocalypse Protocol — "
            "New in Version 2.0 > 7.1 Immediate Actions"
        ],
        "required_facts": [["disconnect"], ["whiteboard"]],
        "stale_facts": [],
    },
    {
        "id": "compare-video-game",
        "question": (
            "What changed in video game time between versions of "
            "the Time and Usage Policy?"
        ),
        "route": "compare",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|2.0|3. Video Game Time > 3.1 Daily Allowance",
            "Time & Usage Policy|1.0|3. Video Game Time > 3.1 Daily Allowance",
        ],
        "required_facts": [["45 minutes"], ["unchanged", "remains", "same"]],
        "stale_facts": [],
    },
    {
        "id": "travel-out-of-scope",
        "question": "What is the travel reimbursement limit for international flights?",
        "route": "lookup",
        "expect_status": "not_in_sources",
        "gold_chunks": [],
        "required_facts": [],
        "stale_facts": [],
    },
    {
        "id": "foosball-dispute-current",
        "question": (
            "Under the Time & Usage Policy, how are disputes over "
            "foosball table access resolved?"
        ),
        "route": "lookup",
        "expect_status": "not_in_sources",
        "gold_chunks": [],
        "required_facts": [],
        "stale_facts": ["whoever is currently winning", "forfeit"],
    },
    {
        "id": "foosball-dispute-compare",
        "question": (
            "What happened to the foosball dispute resolution rule "
            "between Time and Usage Policy versions?"
        ),
        "route": "compare",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|1.0|4. Foosball Time > 4.2 Dispute Resolution"
        ],
        "required_facts": [["removed", "no longer", "dropped", "not included"]],
        "stale_facts": [],
    },
    {
        "id": "compare-tokens",
        "question": (
            "How did the token allocation change between versions "
            "of the Time and Usage Policy?"
        ),
        "route": "compare",
        "expect_status": "answered",
        "gold_chunks": [
            "Time & Usage Policy|1.0|5. Token Allocation > 5.1 Allocation Amount",
            "Time & Usage Policy|2.0|6. Token Allocation > 6.1 Allocation Amount",
        ],
        "required_facts": [
            ["1,000,000", "one million"],
            ["500,000", "five hundred thousand"],
        ],
        "stale_facts": [],
    },
    {
        "id": "compare-shelter-duration",
        "question": (
            "How did the time employees must stay sheltered after a nuclear "
            "event change between Preparedness Policy versions?"
        ),
        "route": "compare",
        "expect_status": "answered",
        "gold_chunks": [
            "Preparedness Policy|1.0|4. Nuclear Apocalypse Protocol > "
            "4.2 All-Clear Timing",
            "Preparedness Policy|2.0|4. Nuclear Apocalypse Protocol — "
            "Updated > 4.3 Duration of Sheltering",
        ],
        "required_facts": [
            ["two hours", "2 hours", "two-hour"],
            ["two weeks", "2 weeks", "two-week"],
        ],
        "stale_facts": [],
    },
]
