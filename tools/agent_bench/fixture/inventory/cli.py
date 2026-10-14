import argparse

from .store import Store


def main(argv=None, store=None):
    store = store or Store()
    ap = argparse.ArgumentParser(prog="inventory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("qty", type=int)
    a.add_argument("price", type=float)
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "add":
        store.add(args.name, args.qty, args.price)
    for name in store.names():
        item = store.items[name]
        print(f"{name}\t{item['qty']}\t{item['price']:.2f}")
    return store
