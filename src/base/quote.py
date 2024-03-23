from dataclasses import dataclass
from datetime import datetime


@dataclass
class Quote:
    """
    A data class representing a quote with a date and price.

    Attributes:
    - date (datetime): The date of the quote.
    - price (float): The price associated with the quote.

    """
    date: datetime
    price: float
