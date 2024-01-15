import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from typing import Optional
import framework
import tasks
import torch
import json

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


def register_args(parser: framework.helpers.ArgumentParser):
    tasks.register_args(parser)
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-lr_warmup", default=0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-analysis.enable", default=True)
    parser.add_argument("-embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.variant", default="standard")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.attention_dropout", default=0.0)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-restore_pretrained", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-train_baseline", default=False, help="Train the model on easy task and test on hard,"
                                                               "no masking")
    parser.add_argument("-lr_sched.steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-lr_sched.gamma", default=0.1)
    parser.add_argument("-lr_sched.type", default="step", choice=["step", "cos"])
    parser.add_argument("-optimizer", default="adam", choice=["adam", "adamw", "sgd", "adagrad"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-adam.eps", default=1e-8)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-tied_embedding", default=False)
    parser.add_argument("-max_length_per_batch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-length_bucketed_sampling", default=False)
    parser.add_argument("-eos", default=True)
    parser.add_argument("-sos", default=True)
    parser.add_argument("-speedtest", default="none", choice=["none", "iter"])
    parser.add_argument("-reg", default=1.0)
    parser.add_argument("-test_only", default=False)
    parser.add_argument("-log_grad_norms", default=False)
    parser.add_argument("-n_microbatch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-dump_logs", default=False)
    parser.add_argument("-dump_validation_plots", default=False)
    parser.add_argument("-val_log_details", default=False)
    parser.add_argument("-nan_detect", default=False)


def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="lm",
                                              register_args=register_args, extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=True, restore=restore)

    task = tasks.get_task(helper.args.task)
    task = task(helper)
    return helper, task

def main():
    helper, task = initialize()
    if helper.args.nan_detect:
        torch.autograd.set_detect_anomaly(True)

    if helper.args.restore_pretrained:
        assert not helper.args.train_baseline

        pretrained = os.path.expanduser(helper.args.restore_pretrained)
        if not helper.args.restore_pretrained.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        task.load_weights(pretrained)
        if helper.args.test_pretrained:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})
        print("Done. Skipping training...")
    elif helper.args.test_only:
        res = task.validate()
        helper.log(res)
        print("Validate returned:")
        print(json.dumps(res))
        print("-------------------")
    else:
        if helper.args.train_baseline:
            task.set_baseline_mode()

        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    if helper.args.analysis.enable and not helper.args.train_baseline:
        task.post_train()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
