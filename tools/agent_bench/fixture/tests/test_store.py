from inventory import Store


def test_add_and_total():
    s = Store()
    s.add("pen", 3, 10.0)
    s.add("pen", 2, 12.0)
    assert s.total() == 5 and s.items["pen"]["price"] == 12.0


def test_remove_all():
    s = Store()
    s.add("cup", 1, 50.0)
    s.remove("cup", 1)
    assert s.names() == []
