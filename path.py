import os.path


class Path:

    fmt_encoded = '%s-enc/%s/*.npz'
    fmt_model = '%s-model/%s/*.npz'
    fmt_solve = '%s-solve/%s-%s/*.npz'
    fmt_play = '%s-play/%s-%s/*.npz'

    def __init__(
            self,
            name: str,
            featurization: str,
            solver: str,
            root: str,
            dataset: str='raw'):
        """Initialize the featurization path.

        :param name: Name of scope for trial
        :param featurization: Name of featurization technique
        :param solver: Name of solution technique
        :param root: Root for all data
        :param dataset: Name of dataset to use
        """
        self.name = name
        self.root = root
        self.featurization = featurization
        self.solver = solver
        self.dataset = dataset

        os.makedirs(self.encoded_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.solve_dir, exist_ok=True)
        os.makedirs(self.play_dir, exist_ok=True)

    @property
    def __base_path(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def data(self) -> str:
        return os.path.join(self.root, self.dataset, '*.npz')

    @property
    def encoded(self) -> str:
        return self.fmt_encoded % (self.__base_path, self.featurization)

    @property
    def encoded_dir(self) -> str:
        return os.path.dirname(self.encoded)

    @property
    def model(self) -> str:
        return self.fmt_model % (self.__base_path, self.featurization)

    @property
    def model_dir(self) -> str:
        return os.path.dirname(self.model)

    @property
    def solve(self) -> str:
        return self.fmt_solve % (
            self.__base_path, self.featurization, self.solver)

    @property
    def solve_dir(self) -> str:
        return os.path.dirname(self.solve)

    @property
    def play(self) -> str:
        return self.fmt_play % (
            self.__base_path, self.featurization, self.solver)

    @property
    def play_dir(self) -> str:
        return os.path.dirname(self.play)