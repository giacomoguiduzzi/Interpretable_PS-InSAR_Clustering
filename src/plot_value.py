import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
file_path = '../data/offida_new/results__offida_new__8___EW.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

string_show = "EW"
metrics = 'dunn'

if "UD" in file_path:
    string_show = "UD"
# Ensure that the columns are in the correct data type
df['pfa_scores'] = pd.to_numeric(df['pfa_scores'], errors='coerce')
df['EW_UD_'+metrics] = pd.to_numeric(df['EW_UD_'+metrics], errors='coerce')
df[string_show+'_'+metrics] = pd.to_numeric(df[string_show+'_'+metrics], errors='coerce')

# Sort the DataFrame based on pfa_scores
df_sorted = df.sort_values(by=['pfa_scores'], ascending=[True])
# Create a unique identifier for colors based on use_extra_features and selection_extra_features
df_sorted['color_group'] = df_sorted.apply(
    lambda row: f"{row['pfa_scores']}__{row['use_extra_features']}__{row['selection_extra_features']}", axis=1
)

# Get unique color groups
unique_groups = df_sorted['color_group'].unique()

# Create a color map with a unique color for each group
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_groups)))
color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

# Assign colors to each bar based on their color group
bar_colors = df_sorted['color_group'].map(color_map)

# Create labels for x-ticks in the desired format
x_tick_labels = df_sorted.apply(
    lambda row: f"{row['pfa_scores']}__{row['use_extra_features']}__{row['selection_extra_features']}"
    if pd.notna(row['pfa_scores']) else row['method'],
    axis=1
)

# Create the figure
plt.figure(figsize=(12, 8))

# Create the bar plot for EW_UD, ordered by pfa_scores
plt.bar(df_sorted.index, df_sorted['EW_UD_'+metrics], color=bar_colors, label='EW_UD (time2feat)', alpha=0.6)

# Create the bar plot for EW (after EW_UD)
plt.bar(df_sorted.index, df_sorted[string_show+'_'+metrics], color=bar_colors, label='UD (kshape and kmeans)', alpha=0.6)

# Filter for the kshape and kmeans methods
kmeans_data = df_sorted[df_sorted['method'] == 'kmeans']
kshape_data = df_sorted[df_sorted['method'] == 'kshape']

# Plot EW values for kshape and kmeans separately
if not kmeans_data.empty:
    plt.bar(kmeans_data.index, kmeans_data[string_show+'_'+metrics], color=bar_colors[kmeans_data.index], label=string_show+' (kmeans)', alpha=0.6)

if not kshape_data.empty:
    plt.bar(kshape_data.index, kshape_data[string_show+'_'+metrics], color=bar_colors[kshape_data.index], label=string_show+' (kshape)', alpha=0.6)

# Set labels and title
plt.xlabel('Index (Ordered by PFA Scores)')
plt.ylabel('Values')
plt.title('Bar Plot of EW_UD and '+string_show+' on Metric '+metrics+'. The cluster are '+str(df.iloc[0]['n_clusters_used'])+'. The value of t2f represents (pfa_scores, use_extra_features, selection_extra_features)')
plt.xticks(df_sorted.index, x_tick_labels, rotation=45, ha='right')


# Show the plot
plt.tight_layout()
plt.show()
