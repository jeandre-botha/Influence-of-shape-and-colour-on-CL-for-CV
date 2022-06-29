# Influence-of-shape-and-colour-on-CL-for-CV

# Config file

| Field          | Description                                    | Default |
|----------------|------------------------------------------------|---------|
| epochs         | Number of epochs over which to train the model |   TBD   |
| learning_rate  | Learning rate for the optimizer                |   TBD   |
| batch_size     | Batch size for training data                   |   TBD   |
| momentum       | Momentum term                                  |   TBD   |
| nesterov       | Use nesterov momentum                          |   TBD   |
| weight_decay   | Weight decay rate                              |   TBD   |
| root_path      | path to project root                           |   TBD   |

For Example:
``` json
{
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 0.1,
    "momentum": 0.0,
    "nesterov": true,
    "weight_decay": 1e-5,
    "root_path" : "./"
}
```
# Supported datasets

| Datset Name | Cli option |
|-------------|------------|
| CIFAR 100   | cifar100   |


# Training and testing a model
From the project root, run command:

``` bash
python .\src\cnn -m <model_name> -d <dataset_name> -a both -c <path_to_config>
```

For example:
``` bash
python .\src\cnn -m baseline -d cifar100 -a both -c .\config_default.json 
```
