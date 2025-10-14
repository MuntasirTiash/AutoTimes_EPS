from models import AutoTimes_Llama, AutoTimes_Gpt2, AutoTimes_Opt_1b


class Exp_Basic(object):
    """Base class for experiments.

    This class provides a basic structure for running experiments, including
    building the model, getting the data, and running training, validation,

    Args:
        args (object): An object containing the arguments for the experiment.
    """

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'AutoTimes_Llama': AutoTimes_Llama,
            'AutoTimes_Gpt2': AutoTimes_Gpt2,
            'AutoTimes_Opt_1b': AutoTimes_Opt_1b
        }
        self.model = self._build_model()

    def _build_model(self):
        """Builds the model.

        This method should be implemented by a subclass.
        """
        raise NotImplementedError

    def _get_data(self):
        """Gets the data.

        This method should be implemented by a subclass.
        """
        pass

    def vali(self):
        """Runs validation.

        This method should be implemented by a subclass.
        """
        pass

    def train(self):
        """Runs training.

        This method should be implemented by a subclass.
        """
        pass

    def test(self):
        """Runs testing.

        This method should be implemented by a subclass.
        """
        pass
