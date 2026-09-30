"""Fixed questions with known source chunks and answer phrases."""

CASES = [
    {
        "question": "When employees are eating cake in a shared space, who must be offered a slice?",
        "chunks": [
            "HR Policy|2.0|7. Shared Refrigerator Policy > 7.2 Cake-Sharing Default"
        ],
        "must_contain": ["leader of the other pod", "slice"],
        "must_not_contain": ["do not offer"],
    },
    {
        "question": "How many paid days off does an employee get for adopting a dog?",
        "chunks": ["HR Policy|2.0|5. Pet Adoption Leave > 5.1 Leave Entitlement"],
        "must_contain": [["7 days", "seven days", "7 paid days", "7 day", "seven day"]],
        "must_not_contain": ["unpaid"],
    },
    {
        "question": "Under the HR Policy, how long must an employee wait before pointing out a manager's mistake?",
        "chunks": ["HR Policy|2.0|6. Boss Error Grace Period"],
        "must_contain": ["30 minutes"],
        "must_not_contain": ["immediately"],
    },
    {
        "question": "What clothing does the HR dress code prohibit?",
        "chunks": ["HR Policy|2.0|4. Dress Code > 4.1 Prohibited Attire"],
        "must_contain": ["suits", "ties"],
        "must_not_contain": ["suits are required"],
    },
    {
        "question": "Under the HR Policy, what happens to food left in the shared refrigerator over a weekend?",
        "chunks": [
            "HR Policy|2.0|7. Shared Refrigerator Policy > 7.3 Weekend Abandonment Consequence"
        ],
        "must_contain": ["abandoned", "spoonful"],
        "must_not_contain": ["may keep it"],
    },
    {
        "question": "What is the recommended maximum caffeine per day?",
        "chunks": ["Health Policy|1.0|5. Caffeine Guidelines > 5.1 Daily Limit"],
        "must_contain": ["400 mg"],
        "must_not_contain": ["500 mg"],
    },
    {
        "question": "How many gym sessions per week are employees expected to complete, and how long is each one?",
        "chunks": [
            "Health Policy|1.0|3. Gym Routine Requirements > 3.1 Minimum Requirement"
        ],
        "must_contain": ["three", "45 minutes"],
        "must_not_contain": ["one session"],
    },
    {
        "question": "Under the Health Policy, are interns required to meet the gym-attendance minimum?",
        "chunks": ["Health Policy|1.0|2. Scope"],
        "must_contain": ["interns", "exempt"],
        "must_not_contain": ["interns must attend"],
    },
    {
        "question": "How many minutes of video games may an employee play per workday?",
        "chunks": [
            "Time and Usage Policy|2.0|3. Video Game Time > 3.1 Daily Allowance"
        ],
        "must_contain": ["45 minutes"],
        "must_not_contain": ["90 minutes"],
    },
    {
        "question": "How many tokens does each employee receive at the start of a six-hour cycle?",
        "chunks": [
            "Time and Usage Policy|2.0|6. Token Allocation > 6.1 Allocation Amount"
        ],
        "must_contain": ["500,000"],
        "must_not_contain": ["two million"],
    },
    {
        "question": "What did Time and Usage Policy 1.0 issue for tokens at the start of each cycle?",
        "chunks": [
            "Time and Usage Policy|1.0|5. Token Allocation > 5.1 Allocation Amount"
        ],
        "must_contain": ["1,000,000"],
        "must_not_contain": ["500,000"],
    },
    {
        "question": "If you win a foosball match, what happens to the other player's tokens?",
        "chunks": [
            "Time and Usage Policy|2.0|4. Foosball Time and the Winner-Takes-Tokens Rule > 4.2 Winner-Takes-Tokens Rule"
        ],
        "must_contain": ["remaining token", "winner"],
        "must_not_contain": ["tokens stay with the loser"],
    },
    {
        "question": "Where should employees shelter when a nuclear detonation is imminent?",
        "chunks": [
            "Preparedness Policy|2.0|4. Nuclear Apocalypse Protocol — Updated > 4.1 Shelter Location"
        ],
        "must_contain": ["break room", "refrigerator"],
        "must_not_contain": ["under their desks"],
    },
    {
        "question": "Who gets a hazmat suit in a nuclear emergency?",
        "chunks": [
            "Preparedness Policy|2.0|4. Nuclear Apocalypse Protocol — Updated > 4.2 Hazmat Suit Eligibility"
        ],
        "must_contain": ["top 10", "foosball"],
        "must_not_contain": ["every employee"],
    },
    {
        "question": "What should employees do first in an AI apocalypse?",
        "chunks": [
            "Preparedness Policy|2.0|7. AI Apocalypse Protocol — New in Version 2.0 > 7.1 Immediate Actions"
        ],
        "must_contain": ["disconnect", "whiteboard"],
        "must_not_contain": ["negotiate with"],
    },
    {
        "question": "What changed in video game time between versions of the Time and Usage Policy?",
        "chunks": [
            "Time and Usage Policy|2.0|3. Video Game Time > 3.1 Daily Allowance",
            "Time and Usage Policy|1.0|3. Video Game Time > 3.1 Daily Allowance",
        ],
        "must_contain": ["45 minutes"],
        "must_not_contain": ["reduced to"],
    },
]
