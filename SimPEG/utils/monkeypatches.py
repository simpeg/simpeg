import properties.basic

def equal(self, value_a, value_b):
    try:
        equal = value_a == value_b
    except:
        return False
    if hasattr(equal, '__iter__'):
        return all(equal)
    return equal

properties.basic.GettableProperty.equal = equal
