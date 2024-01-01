"""
import os
import pandas as pd

# Define the parent directory where the output directories are located
parent_directory = 'fine-tuning/Adam'

highest_validation_score = -1
highest_validation_score_directory = ""

# Iterate through the directories and retrieve the highest validation score
for root, dirs, files in os.walk(parent_directory):
    for directory in dirs:
        logs_path = os.path.join(root, directory, 'valid_logs.csv')
        if os.path.exists(logs_path):
            valid_logs_df = pd.read_csv(logs_path)
            validation_scores = valid_logs_df['iou_score'].max()
            if validation_scores > highest_validation_score:
                highest_validation_score = validation_scores
                highest_validation_score_directory = os.path.join(root, directory)

# Calculate mean IoU from the directory with the highest validation score
if highest_validation_score_directory:
    logs_path = os.path.join(highest_validation_score_directory, 'valid_logs.csv')
    valid_logs_df = pd.read_csv(logs_path)
    mean_iou = valid_logs_df['iou_score'].mean()
    print(f"Directory with the Highest Validation Score: {highest_validation_score_directory}")
    print(f"Highest Validation Score: {highest_validation_score}")
    print(f"Mean IoU of the Directory: {mean_iou}")
else:
    print("No directory with valid logs found.")


"""


import os
import pandas as pd

def find_best_iou_directory(parent_directory):
    best_mean_iou = -1
    best_max_iou = -1
    best_iou_directory = ""
    directory_with_max_iou = ""

    for root, dirs, files in os.walk(parent_directory):
        for directory in dirs:
            logs_path = os.path.join(root, directory, 'valid_logs.csv')
            if os.path.exists(logs_path):
                valid_logs_df = pd.read_csv(logs_path)
                mean_iou = valid_logs_df['iou_score'].mean()
                max_iou = valid_logs_df['iou_score'].max()
                
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_iou_directory = os.path.join(root, directory)
                
                if max_iou > best_max_iou:
                    best_max_iou = max_iou
                    directory_with_max_iou = os.path.join(root, directory)

    return best_iou_directory, best_mean_iou, directory_with_max_iou, best_max_iou



#"""
pretrained_parent_directory = 'fine-tuning/Adam'
not_pretrained_parent_directory = 'not_pretrained/Adam'

best_pretrained_directory, best_pretrained_mean_iou, pretrained_directory_max_iou, pretrained_max_iou = find_best_iou_directory(pretrained_parent_directory)
best_not_pretrained_directory, best_not_pretrained_mean_iou, not_pretrained_directory_max_iou, not_pretrained_max_iou = find_best_iou_directory(not_pretrained_parent_directory)

if best_pretrained_directory:
    print(f"Directory with the Best Mean IoU (Pretrained): {best_pretrained_directory}")
    print(f"Best Mean IoU (Pretrained): {best_pretrained_mean_iou}")
    print(f"Directory with the Max IoU (Pretrained): {pretrained_directory_max_iou}")
    print(f"Max IoU (Pretrained): {pretrained_max_iou}")
else:
    print("No directory with valid logs found for pretrained.")

if best_not_pretrained_directory:
    print(f"Directory with the Best Mean IoU (Not Pretrained): {best_not_pretrained_directory}")
    print(f"Best Mean IoU (Not Pretrained): {best_not_pretrained_mean_iou}")
    print(f"Directory with the Max IoU (Not Pretrained): {not_pretrained_directory_max_iou}")
    print(f"Max IoU (Not Pretrained): {not_pretrained_max_iou}")
else:
    print("No directory with valid logs found for not pretrained.")

#"""

"""
multiclass_parent_directory = 'multi_class/Adam'
best_multiclass_directory, best_multiclass_mean_iou, multiclass_directory_max_iou, multiclass_max_iou = find_best_iou_directory(multiclass_parent_directory)

if best_multiclass_directory:
    print(f"Directory with the Best Mean IoU (multiclass): {best_multiclass_directory}")
    print(f"Best Mean IoU (multiclass): {best_multiclass_mean_iou}")
    print(f"Directory with the Max IoU (multiclass): {multiclass_directory_max_iou}")
    print(f"Max IoU (multiclass): {multiclass_max_iou}")
else:
    print("No directory with valid logs found for multiclass.")"""
