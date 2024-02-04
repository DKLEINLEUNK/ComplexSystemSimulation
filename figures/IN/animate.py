import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_custom_legend():
    # Define the colors and their corresponding labels
    colors = ['#008000', '#ffff00', '#ff0000']
    labels = ['Strong', 'Weak', 'Failed']
    
    # Create patch list for legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    
    # Plotting only for the sake of displaying the legend
    fig, ax = plt.subplots()
    for color in colors:
        ax.plot([], [], ' ', color=color)  # Plot empty lines to create a base for the legend
    
    # Create the custom legend with bold title and increased font sizes
    legend = plt.legend(handles=patches, loc='upper left', title='STATUS', fontsize=24)
    
    # Set the title size and make it bold
    legend.get_title().set_fontsize(24)  # Set the font size of the title
    legend.get_title().set_fontweight('bold')  # Make the title bold
    
    # Optional: Hide axes for a cleaner look since we're only interested in the legend
    ax.axis('off')
    
    plt.show()

# Example usage
create_custom_legend()
