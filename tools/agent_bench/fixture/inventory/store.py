"""A tiny in-memory stock list: the fixture project for the agent benchmark (#96)."""


class Store:
    def __init__(self):
        self.items = {}            # name -> {"qty": int, "price": float}

    def add(self, name, qty, price):
        if qty <= 0:
            raise ValueError("qty must be positive")
        item = self.items.setdefault(name, {"qty": 0, "price": price})
        item["qty"] += qty
        item["price"] = price

    def remove(self, name, qty):
        item = self.items.get(name)
        if item is None:
            return
        item["qty"] -= qty
        if item["qty"] <= 0:
            del self.items[name]

    def total(self):
        return sum(i["qty"] for i in self.items.values())

    def names(self):
        return sorted(self.items)
