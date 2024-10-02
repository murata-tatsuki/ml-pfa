class Stats:
    """Container for statistics per object (event or match)
    
    Useful to keep track of quantities per object, and then combine
    for many objects.
    """
    def __init__(self):
        self.d = {}

    def __getitem__(self, key):
        return self.d[key]

    def add(self, key, val):
        """Add single value to a key"""
        val = np.expand_dims(np.array(val), 0)
        if key not in self.d:
            self.d[key] = val
        else:
            self.d[key] = np.concatenate((self.d[key], val))

    def extend(self, other):
        """Extend with another Stats object"""
        for k, v in other.d.items():
            if k in self.d:
                self.d[k] = np.concatenate((self.d[k], v))
            else:
                self.d[k] = v

    def __len__(self):
        for key, val in self.d.items():
            return len(val)

def dump_stats(outfile, stats):
    outfile = _make_parent_dirs_and_format(outfile)
    np.savez(outfile, **stats.d)

def load_stats(outfile):
    stats = Stats()
    stats.d = dict(np.load(outfile))
    return stats
