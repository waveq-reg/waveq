""" grep -rnw . -e "loss = criterion" """
#./docs/overview.rst:34:      loss = criterion(model, batch)
#./docs/criterions.rst:11:  loss = criterion(model, batch)
./fairseq/tasks/multilingual_translation.py:52:                loss = criterion(model_for_lang_pair(lang_pair), batch)
./fairseq/tasks/semisupervised_translation.py:72:                loss = criterion(model_for_lang_pair(lang_pair), batch)

""" grep -rnw . -e "loss.backward()" """
#./docs/tasks.rst:39:        loss.backward()
./fairseq/tasks/multilingual_translation.py:53:                loss.backward()
./fairseq/tasks/semisupervised_translation.py:73:                loss.backward()
./fairseq/optim/fairseq_optimizer.py:81:        loss.backward()
./fairseq/optim/fp16_optimizer.py:115:        loss.backward()
./fairseq/optim/fp16_optimizer.py:331:        loss.backward()

""" grep -rnw . -e "FairseqOptimizer" """
#./docs/optim.rst:14:.. autoclass:: fairseq.optim.FairseqOptimizer
./fairseq/tasks/fairseq_task.py:327:            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
./fairseq/optim/adagrad.py:8:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/adagrad.py:12:class Adagrad(FairseqOptimizer):
./fairseq/optim/fused_lamb.py:6:from fairseq.optim import FairseqOptimizer, register_optimizer
./fairseq/optim/fused_lamb.py:10:class FairseqLAMB(FairseqOptimizer):
./fairseq/optim/adam.py:14:from fairseq.optim import FairseqOptimizer, register_optimizer
./fairseq/optim/adam.py:21:class FairseqAdam(FairseqOptimizer):
./fairseq/optim/adamax.py:9:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/adamax.py:13:class FairseqAdamax(FairseqOptimizer):
./fairseq/optim/nag.py:9:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/nag.py:13:class FairseqNAG(FairseqOptimizer):
./fairseq/optim/lr_scheduler/fairseq_lr_scheduler.py:6:from .. import FairseqOptimizer
./fairseq/optim/lr_scheduler/fairseq_lr_scheduler.py:13:        if not isinstance(optimizer, FairseqOptimizer):
./fairseq/optim/lr_scheduler/fairseq_lr_scheduler.py:14:            raise ValueError('optimizer must be an instance of FairseqOptimizer')
./fairseq/optim/sgd.py:8:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/sgd.py:12:class SGD(FairseqOptimizer):
./fairseq/optim/__init__.py:10:from fairseq.optim.fairseq_optimizer import FairseqOptimizer
./fairseq/optim/__init__.py:16:    'FairseqOptimizer',
./fairseq/optim/__init__.py:24:    base_class=FairseqOptimizer,
./fairseq/optim/adafactor.py:10:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/adafactor.py:14:class FairseqAdafactor(FairseqOptimizer):
./fairseq/optim/fairseq_optimizer.py:11:class FairseqOptimizer(object):
./fairseq/optim/bmuf.py:9:from . import FairseqOptimizer
./fairseq/optim/bmuf.py:12:class FairseqBMUF(FairseqOptimizer):
./fairseq/optim/fp16_optimizer.py:110:        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
./fairseq/optim/fp16_optimizer.py:210:class FP16Optimizer(_FP16OptimizerMixin, optim.FairseqOptimizer):
./fairseq/optim/fp16_optimizer.py:326:        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
./fairseq/optim/fp16_optimizer.py:384:class MemoryEfficientFP16Optimizer(_MemoryEfficientFP16OptimizerMixin, optim.FairseqOptimizer):
./fairseq/optim/adadelta.py:8:from . import FairseqOptimizer, register_optimizer
./fairseq/optim/adadelta.py:12:class Adadelta(FairseqOptimizer):

""" grep -rnw . -e "train_step" """
./tests/test_bmuf.py:47:def train_step(input, target, model, loss_fn, optimizer, **unused):
./tests/test_bmuf.py:71:        train_step(input, target, model, loss_fn, optimizer)
#./docs/overview.rst:25:          task.train_step(batch, model, criterion, optimizer)
#./docs/overview.rst:31:where the default implementation for ``task.train_step`` is roughly::
#./docs/overview.rst:33:  def train_step(self, batch, model, criterion, optimizer, **unused):
./examples/translation_moe/src/translation_moe.py:190:    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
./fairseq/models/nat/iterative_nonautoregressive_transformer.py:66:        model.train_step = getattr(args, "train_step", 4)
./fairseq/models/nat/iterative_nonautoregressive_transformer.py:86:        for t in range(self.train_step):
./fairseq/models/nat/iterative_nonautoregressive_transformer.py:100:            if t < (self.train_step - 1):
./fairseq/models/nat/iterative_nonautoregressive_transformer.py:195:    args.train_step = getattr(args, "train_step", 4)
./fairseq/tasks/translation_lev.py:142:    def train_step(self,
./fairseq/tasks/multilingual_translation.py:264:    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
./fairseq/tasks/semisupervised_translation.py:324:    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
./fairseq/tasks/fairseq_task.py:28:        Whether the logging outputs returned by `train_step` and `valid_step` can
./fairseq/tasks/fairseq_task.py:315:    def train_step(
./fairseq/trainer.py:329:    def train_step(self, samples, raise_oom=False):
./fairseq/trainer.py:371:                    loss, sample_size_i, logging_output = self.task.train_step(
./fairseq/trainer.py:441:                self.task.train_step(
./fairseq_cli/train.py:187:            log_output = trainer.train_step(samples)

""" grep -rnw . -e "setup_model_loss_criterion" """
./tests/test_bmuf.py:27:def setup_model_loss_criterion(args, rank, is_cuda):
./tests/test_bmuf.py:62:    model, loss_fn, optimizer = setup_model_loss_criterion(args, rank, is_cuda)
