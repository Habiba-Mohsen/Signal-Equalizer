class WindowType:
    def __init__(self, labels, num_sliders):
        self.labels = labels
        self.num_sliders = num_sliders

    def __str__(self):
        return f"labels: {self.labels}, num_sliders: {self.num_sliders}"


class Mode:
    def __init__(self, labels, frq_range, num_sliders):
        self.labels = labels
        self.frq_range = frq_range
        self.num_sliders = num_sliders

    def __str__(self):
        return f"labels: {self.labels}, frq_range: {self.frq_range}, num_sliders: {self.num_sliders}"