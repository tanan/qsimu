class Operator:
    coef: complex
    label: str

    def __init__(self, coef: complex, label: str) -> None:
        self.coef = coef
        self.label = label
