import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths
csv_files = ['lids1.csv', 'lids2.csv','lids3.csv', 'lids4.csv','lids5.csv']  # Update with your actual file paths

plt.figure(figsize=(10, 6))  # Initialize the figure

# Iterate over the list of CSV files
for file in csv_files:
    # Read the current CSV file into a DataFrame
    df = pd.read_csv(file) #, usecols=['round', 'accuracy'])
    
    # Plot the data from this DataFrame
    plt.plot(df,  label=file)

# Customize the plot
plt.title('round vs accuracy')
plt.xlabel('round')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend()  # Show legend to identify lines

# Save the figure as a PDF file
plt.savefig('combined_plot.pdf', bbox_inches='tight')

# Display the plot
plt.show()

