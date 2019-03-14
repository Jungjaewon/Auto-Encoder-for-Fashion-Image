import tensorflow as tf
from tensorboardX import SummaryWriter
#http://www.erogol.com/use-tensorboard-pytorch/

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)
    def image_summary(self, tag, images, step):
        """Add image summary. """
        self.writer.add_image(tag,images,step)