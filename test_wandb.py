import wandb

wandb.init(project="tiny-wfm-backbone", name="stage-a-smoke-test")
for step in range(10):
    wandb.log({"dummy_metric": step * 2})
wandb.finish()
print("Done — check wandb.ai for the run.")
