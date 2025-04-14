import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob

# Path to the directory containing JSON results
#folder_path = '/nfs/FM/gongoubo/test_genmini2.0/LLMTest_NeedleInAHaystack/needlehaystack/results'  # Replace with your folder path
#folder_path = '/data2/wuwenxian/code/end2end/gemini/results'
folder_path = '/nfs/FM/gongoubo/test_genmini2.0/NIAH/results'
# Using glob to find all json files in the directory
json_files = glob.glob(f"{folder_path}/*.json")

# List to hold the data
data = []

# Iterating through each file and extract the 3 columns we need
for file in json_files:
    try:
        with open(file, 'r') as f:
            json_data = json.load(f)
            print(json_data)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            #score = float(json_data.get("score", None).strip())
            score = float(json_data.get("score", None))
            # Appending to the list
            data.append({
            "Document Depth": document_depth,
            "Context Length": context_length,
            "Score": score
            })
    except Exception as e:
        print(e)
        print(file)

# Creating a DataFrame
print(data)
df = pd.DataFrame(data)

print (df.head())
print (f"You have {len(df)} rows")

pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
print(pivot_table.iloc[:5, :5])

# Create a custom colormap. Go to https://coolors.co/ and pick cool colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

# Create the heatmap with better aesthetics
plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
sns.heatmap(
    pivot_table,
    # annot=True,
    fmt="g",
    cmap=cmap,
    cbar_kws={'label': 'Score'}
)

# More aesthetics
plt.title("qwen2.5-7B-instruct-1M")
plt.xlabel('Token Limit')  # X-axis label
plt.ylabel('Depth Percent')  # Y-axis label
plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
plt.ylim(1, 10)
plt.tight_layout()  # Fits everything neatly into the figure area

# Show the plot
plt.show()

plt.savefig("save.png")
