"""Example questions that define the two routes.

Keep them short, policy-agnostic, and different from the eval questions so
the eval measures generalization, not memorization.
"""

LOOKUP_EXAMPLES = (
    "How many vacation days do I get?",
    "What is the rule about bringing pets to work?",
    "Who approves extra tokens?",
    "What does the policy say about dress code?",
    "Where should we go during an emergency?",
    "Are interns covered by this rule?",
    "What did the old version say about breaks?",
    "Is there a limit on coffee?",
)
COMPARE_EXAMPLES = (
    "What changed in the dress code?",
    "How is the new version different from the old one?",
    "What's new in the latest policy?",
    "Was anything removed in the update?",
    "Did the rules about breaks get stricter?",
    "Compare the old and new rules for leave.",
    "What is different between version 1.0 and 2.0?",
    "How did that rule change?",
)
