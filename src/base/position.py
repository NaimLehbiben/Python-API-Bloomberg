
class Position:
    """
    A class representing a position in a portfolio.

    Attributes:
    - asset (FinancialAsset): The asset associated with the position.
    - weight (float): The weight of the position in the portfolio.
    - quantity (float): The quantity of the asset held in the position.
    
    """
    def __init__(self, ticker: str, weight: float = 0):
        """
        Initialize a Position object.

        Parameters:
        - crypto (CryptoAsset): The asset associated with the position.
        - weight (float, optional): The weight of the position in the portfolio (default is 0).
        - quantity (float, optional): The quantity of the asset held in the position (default is 0).
        """
        self.ticker = ticker
        self.weight = weight

