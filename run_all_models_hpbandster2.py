import json
import subprocess

# Load the JSON data
with open('finetune-config_multiclass1.json', 'r') as json_file:
    config = json.load(json_file)

# Define variations for encoder_name and loss[name]
encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']

loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

# Loop through each combination of encoder_name and loss[name]
for encoder_name in encoder_variations:
    for loss_name in loss_variations:
        # Update the values in the config
        config['model']['encoder_name'] = encoder_name
        config['loss']['name'] = loss_name

        # Construct the output directory using updated values
        output_directory = config["output_directory"].format(**config)
        print(output_directory)
        # Write the updated config to a temporary JSON file
        with open('temp_config2.json', 'w') as temp_file:
            json.dump(config, temp_file)

        # Execute the Python file with subprocess module
        subprocess.run(['python', 'hpbandster2.py', '--config', 'temp_config2.json'])
        

# After looping through all combinations, you might want to delete the temporary config file
import os
os.remove('temp_config2.json')
