class Position:
    """
    Une classe représentant une position dans un portefeuille.

    Attributs:
    - ticker (str): Le symbole boursier de l'actif associé à la position.
    - weight (float): La pondération de la position dans le portefeuille.
    """
    def __init__(self, ticker: str, weight: float = 0):
        """
        Initialise un objet Position.

        Args:
            ticker (str): Le symbole boursier de l'actif associé à la position.
            weight (float, optional): La pondération de la position dans le portefeuille (la valeur par défaut est 0).

        Retourne:
            None
        """
        self.ticker = ticker
        self.weight = weight
