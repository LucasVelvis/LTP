"""
Functionalities + dictionaries to extract fallacies from a text.
"""

import re

NON_FALLACIES_REGEX = r"\b(?:does not contain|no|none|not|false|nothing|is not part|not necessarily part|no fallacious|not fallacious)\b"

NUMERIC_TO_LEVEL_2 = {
    0: "nothing",
    1: "appeal to positive emotion",
    2: "appeal to anger",
    3: "appeal to fear",
    4: "appeal to pity",
    5: "appeal to ridicule",
    6: "appeal to worse problems",
    7: "causal oversimplification",
    8: "circular reasoning",
    9: "equivocation",
    10: "false analogy",
    11: "false causality",
    12: "false dilemma",
    13: "hasty generalization",
    14: "slippery slope",
    15: "straw man",
    16: "fallacy of division",
    17: "ad hominem",
    18: "ad populum",
    19: "appeal to (false) authority",
    20: "appeal to nature",
    21: "appeal to tradition",
    22: "guilt by association",
    23: "tu quoque",
    24: "unknown",
}

KEYWORDS_LEVEL_2_NUMERIC = {
    "emotion": 1,
    "anger": 2,
    "fear": 3,
    "pity": 4,
    "ridicule": 5,
    "worse": 6,
    "problems": 6,
    "oversimplification": 7,
    "circular": 8,
    "equivocation": 9,
    "analogy": 10,
    "causality": 11,
    "dilemma": 12,
    "generalization": 13,
    "slippery": 14,
    "slope": 14,
    "straw": 15,
    "division": 16,
    "hominem": 17,
    "populum": 18,
    "authority": 19,
    "nature": 20,
    "tradition": 21,
    "association": 22,
    "quoque": 23,
}


COMBINED_LEVEL_2 = {
    "appeal to positive emotion": ["emotion"],
    "appeal to anger": ["anger"],
    "appeal to fear": ["fear"],
    "appeal to pity": ["pity"],
    "appeal to ridicule": ["ridicule"],
    "appeal to worse problems": ["worse", "problems"],
    "causal oversimplification": ["oversimplification"],
    "circular reasoning": ["circular"],
    "equivocation": ["equivocation"],
    "false analogy": ["analogy"],
    "false causality": ["causality"],
    "false dilemma": ["dilemma"],
    "hasty generalization": ["generalization"],
    "slippery slope": ["slippery", "slope"],
    "straw man": ["straw"],
    "fallacy of division": ["division"],
    "ad hominem": ["hominem"],
    "ad populum": ["populum"],
    "appeal to (false) authority": ["authority"],
    "appeal to nature": ["nature"],
    "appeal to tradition": ["tradition"],
    "guilt by association": ["association"],
    "tu quoque": ["quoque"],
}

def extract_fallacies(text):
    """
    Extract fallacies from the given text. (based on the MAFALDA paper, which provided the dictionaries and the regex pattern)
    """
    non_fallacy_pattern = re.compile(NON_FALLACIES_REGEX, re.IGNORECASE)

    # Store the found/not found fallacies
    fallacies_found = []
    fallacy_found = False

    # Loop through the fallacies and their keywords and check if they are present in the text
    for fallacy, keywords in COMBINED_LEVEL_2.items():
        for keyword in keywords:
            regex_pattern = r'\b{}\b'.format(keyword)
            for match in re.finditer(regex_pattern, text, re.IGNORECASE):
                
                # If match is found also attempt to extract the indices of the location of the fallacy with pattern \d+
                index_pattern = r'\d+'
                indices = re.findall(index_pattern, text[match.end():], re.IGNORECASE)

                # We store not found indices as -1 to indicate that we did not find the indices
                if len(indices) < 2:
                    indices += [-1] * (2 - len(indices))

                # Store fallacy
                fallacies_found.append([int(indices[0]), int(indices[1]), fallacy])
                fallacy_found = True

    # If no fallacies are found, check for non-fallacies
    if not fallacy_found:
        if non_fallacy_pattern.search(text):
            fallacies_found.append([0, 0, "Nothing"])

    return fallacies_found