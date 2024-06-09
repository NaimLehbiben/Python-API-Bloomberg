from dataclasses import dataclass
from datetime import datetime

@dataclass
class Quote:
    """
    Une classe de données représentant une cotation avec une date et un prix.

    Attributs:
    - date (datetime): La date de la cotation.
    - price (float): Le prix associé à la cotation.
    """
    date: datetime
    price: float
