import wandb
import os

def wandb_setup(args):
    os.environ["WANDB_API_KEY"] = args['wandbkey']
    wandb.init(project=args['project'],
               entity=args['wandb_entity'],
               group=args['group'],
               name=args['name'],
               settings=wandb.Settings(start_method='thread'),
               config=args)