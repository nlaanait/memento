import os
from contextlib import redirect_stdout
from spinup.utils.logx import EpochLogger

class VPGLogger(EpochLogger):
    """Logger for Vanilla Policy Gradient (Subclasses spinup.EpochLogger)
    
    Arguments:
        EpochLogger {spinup.utils.logx.EpochLogger} -- spinup's logger
    """
    def __init__(self, *args, **kwargs):
        try:
            self.dump = kwargs.pop('config_dump')
        except KeyError:
            self.dump = False
        super(VPGLogger, self).__init__(*args, **kwargs)

    def save_config(self, config):
        if self.dump:
            with redirect_stdout(None):
                super(VPGLogger, self).save_config(config)
        
    def _pytorch_simple_save(self, itr=None):
        if os.path.exists(self.output_dir):
            fpath = os.path.join(self.output_dir, 'actor_critic_model.pt')
            torch.save(self.pytorch_saver_elements.state_dict(), fpath)


