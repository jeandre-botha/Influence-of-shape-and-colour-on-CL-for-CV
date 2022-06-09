# Influence-of-shape-and-colour-on-CL-for-CV

# Config file

| Field          | Description                                    | Default |
|----------------|------------------------------------------------|---------|
| epochs         | Number of epochs over which to train the model |   TBD   |
| learning_rate  | Learning rate for the optimizer                |   TBD   |
| batch_size     | Batch size for training data                   |   TBD   |

For Example:
``` json
{
    "batch_size": 64,
    "epochs": 2,
    "learning_rate": 0.001
}
```
# Supported datasets

| Datset Name | Cli option |
|-------------|------------|
| CIFAR 100   | cifar100   |


# Training a model
From the project root, run command:

``` bash
python .\src\cnn -m <model_name> -d <dataset_name> -a train -c <path_to_config>
```

For example:
``` bash
python .\src\cnn -m baseline -d cifar100 -a train -c .\config_default.json 
```