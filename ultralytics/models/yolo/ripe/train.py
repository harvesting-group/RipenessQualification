# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import RipenessModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results, plot_images_ripeness

#
class RipenessTrainer(yolo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a RipenessTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'ripe' #'segment' #
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = RipenessModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'ripe_loss'
        return yolo.ripe.RipenessValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images_ripeness(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    #batch['ripeness'].squeeze(-1),# insert
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)
        # plot_images(batch["img"],
        #             batch["batch_idx"],
        #             batch["cls"].squeeze(-1),
        #             batch["bboxes"],
        #             masks=batch["masks"],
        #             paths=batch["im_file"],
        #             fname=self.save_dir / f"train_batch{ni}.jpg",
        #             on_plot=self.on_plot)


    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
