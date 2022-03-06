from .trainer import Trainer
import wandb


def tune_hyperparameter(
        train_dataset, val_dataset,  # data loader
        hyper_params_setting, default_setting,  # hyperparameters for tuning
        project, entity,  # names
        arguments,
        train_target_format=lambda x: x, train_output_format=lambda x: x,
        val_target_format=lambda x: x, val_output_format=lambda x: x
        # function that transforms a variable into a format
):
    '''
    hyper_params_setting example)
    hyper_params_setting = {
        'method': 'grid',  # grid, random
        'metric': {
            'name': 'Test_Loss',
            'goal': 'minimize'
        },
        'parameters': {
            'model': {
                'values': [resnet18(), resnet34()]
            },
            'batch_size': {
                'values': [4, 8, 16]
            },
            'learning_rate': {
                'values': [0.00001, 0.00003, 0.0001, 0.0003, 0.001]
            },
            'weight_decay': {
                'values': [1e-5, 5e-5, 1e-4]
            },
            'criterion': {
                'values': [nn.CrossEntropyLoss()]
            },
            'optimizer': {
                'values': [lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
                           lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd)]
            }
        }
    }
    '''

    models = [model for model in hyper_params_setting['parameters']['model']['values']]
    criterions = [criterion for criterion in hyper_params_setting['parameters']['criterion']['values']]
    optimizers = [optimizer for optimizer in hyper_params_setting['parameters']['optimizer']['values']]
    hyper_params_setting['parameters']['model']['values'] = list(range(len(models)))
    hyper_params_setting['parameters']['criterion']['values'] = list(range(len(criterions)))
    hyper_params_setting['parameters']['optimizer']['values'] = list(range(len(optimizers)))

    wandb.login()
    sweep_id = wandb.sweep(hyper_params_setting, project=project, entity=entity)

    def train():

        wandb.init(config=default_setting)
        hyper_params = wandb.config

        model = models[hyper_params['model']].to(arguments['device'])
        batch_size = hyper_params['batch_size']
        lr = hyper_params['learning_rate']
        weight_decay = hyper_params['weight_decay']
        criterion = criterions[hyper_params['criterion']]
        optimizer = optimizers[hyper_params['optimizer']]

        train_loader = train_dataset(batch_size)
        val_loader = val_dataset(batch_size)

        min_val_loss = float('inf')
        min_val_step = 0

        trainer = Trainer(
            model=model, criterion=criterion, optimizer=optimizer(model.parameters(), lr=lr, wd=weight_decay),
            train_dataloader=train_loader, val_dataloader=val_loader,
            train_target_format=train_target_format, train_output_format=train_output_format,
            val_target_format=val_target_format, val_output_format=val_output_format,
            params=arguments
        )

        for epoch in range(500):
            train_metric = trainer.train_epoch(epoch)
            val_metric = trainer.validation_epoch(epoch)

            train_loss = train_metric.metrics['Loss']['avg']
            val_loss = val_metric.metrics['Loss']['avg']

            # Early Stopping
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                min_val_step = epoch
                # save model parameter
                # torch.save(model.state_dict(), arguments['save_directory'] + f'/best_model_{ith_params}.pt')

            if epoch - min_val_step > arguments['early_stopping_step']:
                break

            wandb.log({'Val_Loss': val_loss, 'Min_Val_Loss': min_val_loss, 'Loss': train_loss})

    wandb.agent(sweep_id, train)

    return sweep_id