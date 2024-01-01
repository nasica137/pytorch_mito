import os
import pandas as pd

# Define the parent directories where the output directories are located
parent_directories = ['multi_class_hpbandster', 'multi_class_hpbandster_not_pretrained']

# Create a dictionary to store the top 5 subdirectories for each metric
top_subdirectories = {
    'Best Mean IoU': {},
    'Best Mean Loss': {},
    'Best Max IoU': {}
}


for directory in parent_directories:
    top_subdirectories['Best Mean IoU'][directory] = []
    top_subdirectories['Best Mean Loss'][directory] = []
    top_subdirectories['Best Max IoU'][directory] = []

    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            # Read the validation logs from the valid_logs.csv file
            logs_path = os.path.join(directory, subdir, 'valid_logs.csv')
            if os.path.exists(logs_path):
                valid_logs_df = pd.read_csv(logs_path)
                
                # Calculate mean IoU and mean loss
                mean_iou = round(valid_logs_df['iou_score'].mean(), 3)
                mean_loss = round(valid_logs_df.iloc[:, 1:].mean().values[0], 3)  # Change the index if needed
                
                # Find the top 5 subdirectories for each metric
                top_subdirectories['Best Mean IoU'][directory].append((subdir, mean_iou))
                top_subdirectories['Best Mean Loss'][directory].append((subdir, mean_loss))
                top_subdirectories['Best Max IoU'][directory].append((subdir, round(valid_logs_df['iou_score'].max(), 3)))

    # Sort the top subdirectories based on each metric
    for metric in top_subdirectories:
        top_subdirectories[metric][directory] = sorted(top_subdirectories[metric][directory], key=lambda x: x[1], reverse=True)[:5]

# Save the top 5 subdirectories for each metric in each parent directory to CSV files
for directory in parent_directories:
    for metric, values in top_subdirectories.items():
        df = pd.DataFrame(values[directory], columns=['Subdirectory', 'Score'])
        df['Score'] = df['Score'].round(3)  # Round scores to 3 decimal places
        filename = f'plots/evaluate_{directory}/{directory}_top5_{metric.replace(" ", "_")}.csv'
        df.to_csv(filename, index=False)




# max iou

import pandas as pd

# Load the CSV files containing the top 5 subdirectories based on Max IoU
filename_pretrained = 'plots/evaluate_multi_class_hpbandster/multi_class_hpbandster_top5_Best_Max_IoU.csv'
filename_not_pretrained = 'plots/evaluate_multi_class_hpbandster_not_pretrained/multi_class_hpbandster_not_pretrained_top5_Best_Max_IoU.csv'

# Load dataframes from both CSV files
df_pretrained = pd.read_csv(filename_pretrained)
df_not_pretrained = pd.read_csv(filename_not_pretrained)

# Get the row with the highest Max IoU score for pretrained models
highest_pretrained = df_pretrained.loc[df_pretrained['Score'].idxmax()]

# Get the row with the highest Max IoU score for not_pretrained models
highest_not_pretrained = df_not_pretrained.loc[df_not_pretrained['Score'].idxmax()]

# Combine both highest rows into a single dataframe
combined_highest_df = pd.concat([highest_pretrained, highest_not_pretrained], axis=1).T

# Add a column indicating whether the model is pretrained or not_pretrained
combined_highest_df['Pretrained'] = ['Yes', 'No']

# Save the combined highest values to a new CSV file
output_highest_filename = 'plots/evaluate_both_transfer_learning/highest_models_max_iou.csv'
combined_highest_df.to_csv(output_highest_filename, index=False)


# mean iou


# Load the CSV files containing the top 5 subdirectories based on Max IoU
filename_pretrained = 'plots/evaluate_multi_class_hpbandster/multi_class_hpbandster_top5_Best_Mean_IoU.csv'
filename_not_pretrained = 'plots/evaluate_multi_class_hpbandster_not_pretrained/multi_class_hpbandster_not_pretrained_top5_Best_Mean_IoU.csv'

# Load dataframes from both CSV files
df_pretrained = pd.read_csv(filename_pretrained)
df_not_pretrained = pd.read_csv(filename_not_pretrained)

# Get the row with the highest Mean IoU score for pretrained models
highest_iou_pretrained = df_pretrained.loc[df_pretrained['Score'].idxmax()]

# Get the row with the highest Mean IoU score for not_pretrained models
highest_iou_not_pretrained = df_not_pretrained.loc[df_not_pretrained['Score'].idxmax()]

# Combine both highest IoU rows into a single dataframe
combined_highest_iou_df = pd.concat([highest_iou_pretrained, highest_iou_not_pretrained], axis=1).T

# Add a column indicating whether the model is pretrained or not_pretrained
combined_highest_iou_df['Pretrained'] = ['Yes', 'No']

# Save the combined highest IoU values to a new CSV file
output_highest_iou_filename = 'plots/evaluate_both_transfer_learning/highest_models_mean_iou.csv'
combined_highest_iou_df.to_csv(output_highest_iou_filename, index=False)


# loss

# Load the CSV files containing the top 5 subdirectories based on Max IoU
filename_pretrained = 'plots/evaluate_multi_class_hpbandster/multi_class_hpbandster_top5_Best_Mean_Loss.csv'
filename_not_pretrained = 'plots/evaluate_multi_class_hpbandster_not_pretrained/multi_class_hpbandster_not_pretrained_top5_Best_Mean_Loss.csv'

# Load dataframes from both CSV files
df_pretrained = pd.read_csv(filename_pretrained)
df_not_pretrained = pd.read_csv(filename_not_pretrained)


# Get the row with the lowest (best) Mean Loss for pretrained models
lowest_loss_pretrained = df_pretrained.loc[df_pretrained['Score'].idxmin()]

# Get the row with the lowest (best) Mean Loss for not_pretrained models
lowest_loss_not_pretrained = df_not_pretrained.loc[df_not_pretrained['Score'].idxmin()]

# Combine both lowest Loss rows into a single dataframe
combined_lowest_loss_df = pd.concat([lowest_loss_pretrained, lowest_loss_not_pretrained], axis=1).T

# Add a column indicating whether the model is pretrained or not_pretrained
combined_lowest_loss_df['Pretrained'] = ['Yes', 'No']

# Save the combined lowest Loss values to a new CSV file
output_lowest_loss_filename = 'plots/evaluate_both_transfer_learning/lowest_models_mean_loss.csv'
combined_lowest_loss_df.to_csv(output_lowest_loss_filename, index=False)
