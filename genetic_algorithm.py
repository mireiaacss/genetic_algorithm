import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os
import random
import math
from PIL import Image
from scipy.spatial import KDTree
from loguru import logger

################### SESSION 1 ###################

# Function 1: Download MNIST images and save them locally
def download_mnist_images():
    # Download MNIST dataset and store images and labels
    # Save the first 500 images and their labels to a folder

    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False) #fetch the MNIST dataset
    
    selected_images = mnist.data[:500] #select the first 500 images
    selected_labels = mnist.target[:500] #select the first 500 labels
    if not os.path.exists("mnist_images"): #check if the folder exists
        os.makedirs('mnist_images') #create a folder to save the images if the folder does not exist

    for i, (image, label) in enumerate(zip(selected_images, selected_labels)): #loop through the selected images and labels
        img_array = image.reshape(28, 28) #reshape the image to 28x28
        img = Image.fromarray(img_array.astype(np.uint8)) #convert the image to a PIL image
        file_name = f"mnist_images/img_{i}_label{label}.png" #create a file name for the image
        img.save(file_name) #save the image to the folder

    logger.success("MNIST images saved to 'mnist_images' folder.") #log the success of the function
    return selected_images, selected_labels #return the selected images and labels
    

# Function 2: Analyze the images
def analyze_images(images):
    # Print image details: number, dimensions, pixel count, and range
    # Check if image is PNG (alpha channel)
    
    for i, image_data in enumerate(images): #loop through the images
        print(f"\n ---Image{i} ---") #print the image number
        print(f"Dimensions: {image_data.shape}") #print the dimensions of the image
        print(f"Pixel_count: {image_data.size} ") #print the pixel count of the image
        print(f"Range: ({image_data.min()}, {image_data.max()})") #print the range of the image
        
        has_alpha = False #initialise the has_alpha variable
        img_mode = None #initialise the img_mode variable

        if image_data.ndim == 3 and image_data.shape[-1] == 4: #check if the image has an alpha channel
            has_alpha = True  #set the has_alpha variable to True
            img_mode = "Implied RGBA (from shape)" #set the img_mode variable to "Implied RGBA (from shape)"
        
        elif image_data.ndim == 3 and image_data.shape[-1] == 2: #check if the image has an alpha channel
            has_alpha = True  #set the has_alpha variable to True
            img_mode = "Implied LA (from shape)" #set the img_mode variable to "Implied LA (from shape)"

        if img_mode: #if the img_mode variable is not None
            print(f"Interpreted Mode: {img_mode}") #print the img_mode variable
        
        print(f"Has Alpha Channel (Proxy for PNG Transparency): {has_alpha}") #print the has_alpha variable


# Function 3: Crop the image identifying empty rows and columns
def crop_image(image):
    # Identify non-zero rows and columns
    # Crop the image to the non-zero region and pad to 28x28 if needed

    non_empty_rows = np.any(image > 0, axis=1) #identify the non-zero rows
    non_empty_columns = np.any(image > 0, axis=0) #identify the non-zero columns

    row_indices = np.where(non_empty_rows)[0] #identify the indices of the non-zero rows
    columns_indices = np.where(non_empty_columns)[0] #identify the indices of the non-zero columns

    if len(row_indices) == 0 or len(columns_indices) == 0: #if the row_indices or columns_indices are empty
        return image #return the image

    top, bottom = row_indices[0], row_indices[-1] + 1 #identify the top and bottom of the image
    left, right = columns_indices[0], columns_indices[-1] + 1 #identify the left and right of the image

    cropped = image[top:bottom, left:right] #crop the image to the non-zero region
    
    return cropped #return the cropped image


# Function 4: Plot histogram comparison
def plot_histogram_comparison(images, labels, max_labels_per_figure=5):
    # Select first occurrences of each label and compute histograms
    # Plot histograms in multiple figures if needed

    if images.size == 0 or labels.size == 0: #if the images or labels are empty
        logger.warning("Empty images or labels provided for histogram comparisons.") #log the warning
        return  #return None
    
    first_occurence_indices = {} #initialise the first_occurence_indices dictionary, that will store the first occurence of each label
    unique_labels_found = [] #initialise the unique_labels_found list, that will store the unique labels

    for index, label in enumerate(labels): #loop through the labels
        label_str = str(label) #convert the label to a string to assure the consistency of the keys
        if label_str not in first_occurence_indices: #if the label is not in the first_occurence_indices dictionary
            first_occurence_indices[label_str] = index #add the label to the first_occurence_indices dictionary
            unique_labels_found.append(label_str) #add the label to the unique_labels_found list
    
    try:
        sorted_unique_labels = sorted(unique_labels_found, key=lambda x: int(x)) #sort the unique labels numerically
    except ValueError: #if the labels could not be sorted numerically
        logger.warning("Labels could not be sorted numerically, using string sort.") #log the warning
        sorted_unique_labels = sorted(unique_labels_found) #sort the unique labels stringly
    
    if not sorted_unique_labels: #if the unique labels are not found
        logger.warning("No unique labels found.") #log the warning

    logger.success(f"Found unique labels: {', '.join(sorted_unique_labels)}") #log the success of the function

    num_labels = len(sorted_unique_labels) #identify the number of labels
    num_figures = math.ceil(num_labels / max_labels_per_figure) #identify the number of figures to plot the histograms
    logger.info(f"Plotting {num_labels} histograms across {num_figures} figure(s)") #log the number of histograms and figures

    epsilon = np.exp(-20) #Set epsilon to a small value to avoid log(0)
    for fig_idx in range(num_figures): #loop through the figures

        start_index = fig_idx * max_labels_per_figure #identify the start index of the figure
        end_index = min(start_index + max_labels_per_figure, num_labels) #identify the end index of the figure
        num_plots_on_this_figure = end_index - start_index #identify the number of plots on this figure 

        if num_plots_on_this_figure <= 0: #if the number of plots on this figure is less than or equal to 0
            continue #continue to the next figure
            
        if num_plots_on_this_figure == 1: #if the number of plots on this figure is 1
            cols = 1 #identify the number of columns
            rows = 1 #identify the number of rows
        elif num_plots_on_this_figure <= 4: #if the number of plots on this figure is less than or equal to 4
            cols = 2 #identify the number of columns
            rows = math.ceil(num_plots_on_this_figure / 2) #identify the number of rows
        else: #if the number of plots on this figure is greater than 4
            cols = 3 #identify the number of columns
            rows = math.ceil(num_plots_on_this_figure / cols) #identify the number of rows
        
        _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False) #create the figure and axes
        plot_counter = 0 #initialise the plot counter
        axes_flat = axes.flatten() #flatten the axes
        
        for i in range(start_index, end_index): #loop through the plots
            label = sorted_unique_labels[i] #identify the label
            image_index = first_occurence_indices[label] #identify the image index

            image_data = images[image_index] #identify the image data
            image_flat = image_data.flatten() #flatten the image data

            total_pixels = image_flat.size #identify the total number of pixels
            if total_pixels == 0: #if the total number of pixels is 0
                logger.warning(f"Image: {image_index} for label '{label}' has zero pixels. Skipping histogram.") #log the warning

            hist, bin_edges = np.histogram(image_flat, bins=50, range=(0, 256)) #create the histogram
            
            log_hist = np.log(hist.astype(np.float64) + epsilon) #identify the log of the histogram

            ax = axes_flat[plot_counter] #identify the axis
            ax.plot(bin_edges[:-1], log_hist, label=f"Digit '{label}'", color='darkturquoise', linewidth=1.0) #plot the histogram
            ax.set_title(f"Log-Normalised Histogram for Label {label}") #set the title of the histogram
            ax.set_xlabel("Pixel Intensity") #set the x label of the histogram
            ax.set_ylabel("Log of frequency") #set the y label of the histogram
            ax.grid(True, linestyle='--', alpha = 0.6) #set the grid of the histogram

            plot_counter += 1 #increment the plot counter


        for k in range(plot_counter, len(axes_flat)): #loop through the axes    
            axes_flat[k].axis('off') #turn off the axis

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) #tight layout the plot

    
    plt.show() #show the plot




################### SESSION 2 ###################


def classify_images_with_kdtree(images, labels):
    
    # Build KDTree
    if images.ndim > 2: #if the images are not flattened
        flattened_images = np.array([img.flatten() for img in images]) #flatten the images
    else:
        flattened_images = images #if the images are flattened, assign the images to flattened_images
    
    if not flattened_images.size: #if the flattened_images are empty
        logger.warning("No images to classify with improved KDTree.") #log the warning
        return np.array([]) #return an empty array
    
    kdtree = KDTree(flattened_images) #create a KDTree
    # Define a classification rule based on distances (ignore self-distance)
    distance, _ = kdtree.query(flattened_images, k=2) #query the KDTree
    nearest_distances = distance[:, 1] #identify the nearest distances, while ignoring the self-distance
    # Thresholding based on median distance
    median_distance = np.median(nearest_distances) #identify the median distance
    classification = np.zeros(len(images), dtype=int) #initialise the classification array
    classification[nearest_distances < median_distance] = 1 #set the classification to 1 if the nearest distance is less than the median distance
    
    logger.info("Finished KDTree classification") #log the success of the function
    return classification #return the classification

def classify_images_with_kdtree_improved(images, labels):
    """
    Improvement:
    - Continous score instead of binary
    - Clasifier only compares images of the same class
    """
    
    # Build KDTree
    if images.ndim > 2: #if the images are not flattened
        flattened_images = np.array([img.flatten() for img in images]) #flatten the images
    else:
        flattened_images = images #if the images are flattened, assign the images to flattened_images
    
    if not flattened_images.size: #if the flattened_images are empty
        logger.warning("No images to classify with improved KDTree.") #log the warning
        return np.array([]) #return an empty array
    
    n = len(flattened_images) #identify the number of images
    classification = np.zeros(n, dtype=float) #initialise the classification array

    for cls in np.unique(labels): #loop through the unique labels
        idx = np.where(labels == cls)[0] #identify the indices of the unique labels
        if len(idx) < 2: #if the number of indices is less than 2
            classification[idx] = 1.0 #set the classification to 1.0 as there is no neighbour to compare against
            continue #continue to the next label
        
        data = flattened_images[idx] #identify the data
        kdtree = KDTree(data) #create a KDTree
        distances, _ = kdtree.query(data, k=2) #query the KDTree
        nearest_distance = distances[:, 1] #identify the nearest distance, while ignoring the self-distance

        d_min, d_max = nearest_distance.min(), nearest_distance.max() #identify the minimum and maximum of the nearest distance
        if d_max - d_min < 1e-12: #if the maximum and minimum of the nearest distance are the same
            similarity = np.ones_like(nearest_distance) #set the similarity to 1
        else: #if the maximum and minimum of the nearest distance are not the same
            similarity = 1.0 - (nearest_distance - d_min) / (d_max - d_min) #set the similarity to the continous score
        
        classification[idx] = similarity #set the classification to the similarity
    
    logger.success("Finished continous same-class KDTree classification") #log the success of the function
    return classification #return the classification





def create_random_individual(images, individual_length, labels, kdtree_vector):
    # Select random indexes
    indexes = [] #initialise the indexes array
    for _ in range(individual_length): #loop through the individual length
        random_index = np.random.randint(0, len(images) - 1) #select a random index
        indexes.append(random_index) #append the random index to the indexes array

    return images[indexes], labels[indexes], kdtree_vector[indexes] #return the images, labels and kdtree vector


def selection(population, fitness_values):
    # Select the top 2 individuals based on fitness

    parent_1 = population[fitness_values[0]] #identify the first parent
    parent_2 = population[fitness_values[1]] #identify the second parent

    return (parent_1, parent_2) #return the parents

def crossover(parent1, parent2, individual_length):
    
    # Split parents into images, labels and kdtree
    parent1_images = parent1[0] #identify the first parent's images
    parent1_labels = parent1[1] #identify the first parent's labels
    parent1_kdtree = parent1[2] #identify the first parent's kdtree

    parent2_images = parent2[0] #identify the second parent's images
    parent2_labels = parent2[1] #identify the second parent's labels
    parent2_kdtree = parent2[2] #identify the second parent's kdtree

    # Perform crossover separately for images, labels and kdtree
    random_index = np.random.randint(0, individual_length - 1) #select a random index
    child_images = np.concatenate((parent1_images[:random_index], parent2_images[random_index:])) #concatenate the first parent's images and the second parent's images
    child_labels = np.concatenate((parent1_labels[:random_index], parent2_labels[random_index:])) #concatenate the first parent's labels and the second parent's labels
    child_kdtree = np.concatenate((parent1_kdtree[:random_index], parent2_kdtree[random_index:])) #concatenate the first parent's kdtree and the second parent's kdtree

    return child_images, child_labels, child_kdtree # Return child as a tuple


def mutation(individual, individual_length, images, labels, kdtree_vector):
    
    individual_image, individual_label, individual_kdtree = individual #identify the individual's images, labels and kdtree
    # Select a mutation point randomly
    mutation_point = np.random.randint(0, individual_length) #select a mutation point

    # Select a new random image and corresponding label
    random_index = np.random.randint(0, len(images) - 1) #select a random index
    random_image = images[random_index] #identify the random image
    random_label = labels[random_index] #identify the random label
    
    new_images = individual_image.copy() #copy the individual's images
    new_labels = individual_label.copy() #copy the individual's labels
    new_kdtree = individual_kdtree.copy() #copy the individual's kdtree
    # Calculate new kdtree 
    new_images[mutation_point] = random_image #replace the mutation point with the random image
    new_labels[mutation_point] = random_label #replace the mutation point with the random label
    new_kdtree[mutation_point] = kdtree_vector[random_index] #replace the mutation point with the random kdtree

    
    return new_images, new_labels, new_kdtree #New individual


def fitness(individual, pi_digits):
    # Calculate total fitness_value based on pi_quality (total correct matches with the image labels) + kdtree_quality (sum of kdtree_vector)
    _, individual_label, individual_kdtree = individual #identify the individual's labels and kdtree
    pi_quality = 0 #initialise the pi quality

    for i in range(len(individual_label)): #loop through the individual's labels
        if str(individual_label[i]) == str(pi_digits[i]): #if the individual's label is equal to the pi digits
            pi_quality += 1 #increment the pi quality
    
    kdtree_quality = np.sum(individual_kdtree) #identify the kdtree quality

    fitness_value = pi_quality + kdtree_quality #identify the fitness value



    return fitness_value, pi_quality, kdtree_quality #return the fitness value, pi quality and kdtree quality


def fitness_improved(individual, pi_digits):
    """
    Improvement:
    - Weighted kdtree quality
    """
    # Calculate total fitness_value based on pi_quality (total correct matches with the image labels) + kdtree_quality (sum of kdtree_vector)
    _, individual_label, individual_kdtree = individual #identify the individual's labels and kdtree
    pi_quality = 0 #initialise the pi quality

    for i in range(len(individual_label)): #loop through the individual's labels
        if str(individual_label[i]) == str(pi_digits[i]): #if the individual's label is equal to the pi digits
            pi_quality += 1 #increment the pi quality
    
    kdtree_quality = np.sum(individual_kdtree) #identify the kdtree quality

    weight_kdtree = 0.5 #set the weight of the kdtree to 0.5
    fitness_value = pi_quality + weight_kdtree * kdtree_quality #identify the fitness value



    return fitness_value, pi_quality, kdtree_quality #return the fitness value, pi quality and kdtree quality





def plot_evolution(best_pi_quality_values, avg_pi_quality_values, best_kdtree_values, avg_kdtree_values):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) #create the figure and axes
    
    generations = range(1, len(best_pi_quality_values) + 1) #identify the generations
    

    ax1.plot(generations, best_pi_quality_values, 'b-', linewidth=2, label='Best pi_quality') #plot the best pi quality
    ax1.plot(generations, avg_pi_quality_values, 'c--', linewidth=1.5, label='Avg pi_quality') #plot the average pi quality
    ax1.set_xlabel('Generation') #set the x label
    ax1.set_ylabel('Fitness (pi_quality)') #set the y label
    ax1.set_title('Fitness Evolution - pi_quality') #set the title
    ax1.grid(False) #turn off the grid
    ax1.legend() #show the legend
    if best_pi_quality_values: #if the best pi quality values are not empty
        ax1.set_ylim(min(0, np.min(avg_pi_quality_values)-1 if avg_pi_quality_values else 0), 
                     max(10.5, np.max(best_pi_quality_values)+0.5 if best_pi_quality_values else 10.5)) #set the y limits


    ax2.plot(generations, best_kdtree_values, 'r-', linewidth=2, label='Best kdtree_vector') #plot the best kdtree vector
    ax2.plot(generations, avg_kdtree_values, color='orange', linestyle='--', linewidth=1.5, label='Avg kdtree_vector') #plot the average kdtree vector
    ax2.set_xlabel('Generation') #set the x label
    ax2.set_ylabel('Fitness (kdtree_vector)') #set the y label
    ax2.set_title('Fitness Evolution - kdtree_vector') #set the title
    ax2.grid(False) #turn off the grid
    ax2.legend() #show the legend
    if best_kdtree_values : #if the best kdtree values are not empty
         ax2.set_ylim(min(4.0, np.min(avg_kdtree_values)-0.5 if avg_kdtree_values else 4.0),
                      max(8.5, np.max(best_kdtree_values)+0.5 if best_kdtree_values else 8.5)) #set the y limits


    plt.tight_layout() #tight layout the plot
    plt.show() #show the plot


def plot_best_individual(best_individual, pi_digits):
  
    best_images, best_labels, _ = best_individual  #identify the best images, labels and kdtree
    pi_digits_for_fitness = [str(p) for p in pi_digits] #identify the pi digits for fitness

    total_fitness_val, pi_quality_val, kdtree_quality_val = fitness(best_individual, pi_digits_for_fitness) #identify the total fitness value, pi quality value and kdtree quality value

    num_images = len(best_images) #identify the number of images
    
    if num_images != 10: #if the number of images is not 10
        print(f"Warning: Expected 10 images for 2x5 plot, got {num_images}. Output may differ from expected.") #print the warning

    cols = 5 #identify the number of columns
    rows = (num_images + cols -1) // cols if num_images > 0 else 1 #identify the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5)) #create the figure and axes
    if num_images == 0: #if the number of images is 0
        axes = np.array([]) #set the axes to an empty array
    elif num_images == 1 and rows == 1 and cols == 1: #if the number of images is 1, the number of rows is 1 and the number of columns is 1
        axes = np.array([axes]) #set the axes to an array
    axes = axes.flatten() #flatten the axes
    
    title_str = f"Best-Fit Individual Images (Fitness value: {total_fitness_val:.0f}, Pi quality: {pi_quality_val:.0f}, KDTree quality: {kdtree_quality_val:.0f})" #identify the title string
    fig.suptitle(title_str, fontsize=12) #set the title of the figure
    
    for i in range(num_images): #loop through the images
        img = best_images[i] #identify the image
        if img.ndim == 1 and img.shape[0] == 28*28: #if the image is 1D and the shape of the image is 28x28
            img = img.reshape(28, 28) #reshape the image to 28x28
        
        axes[i].imshow(img, cmap='gray') #plot the image
        current_pi_digit = pi_digits[i] if i < len(pi_digits) else "N/A" #identify the current pi digit
        axes[i].set_title(f"Label: {best_labels[i]}\nReal: {current_pi_digit}", fontsize=9) #set the title of the image
        axes[i].axis('off') #turn off the axis
       
    
    for i in range(num_images, len(axes)): #loop through the images
        axes[i].axis('off') #turn off the axis
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #tight layout the plot
    plt.show() #show the plot


def genetic_algorithm(images, labels, kdtree_vector, improvements=False):

    # Step 1: Define the algorithm parameters (e.g., population size, number of generations, mutation rate)
    num_generations = 100  # Adjust as needed
    population_size = 50 # Adjust as needed
    individual_length = 10 #identify the individual length
    pi_digits = [1,4,1,5,9,2,6,5,3,5] #identify the pi digits
    mutation_rate = 0.2 #identify the mutation rate
    
    best_pi_quality_values = [] #initialise the best pi quality values
    avg_pi_quality_values = [] #initialise the average pi quality values
    best_kdtree_values = [] #initialise the best kdtree values
    avg_kdtree_values = [] #initialise the average kdtree values

    # Step 2: Initialize the population with randomly selected individuals
    current_population = [] #initialise the current population
    for _ in range(population_size): #loop through the population size
        random_individual = create_random_individual(images, individual_length, labels, kdtree_vector) #create a random individual
        current_population.append(random_individual) #append the random individual to the current population

    
    # Step 3: Evolve the population over multiple generations
    for _ in range(num_generations): #loop through the number of generations
        # Step 3.1: Compute the fitness score for each individual
        fitness_score = {} #initialise the fitness score
        pi_qualities = [] #initialise the pi qualities
        kdtree_qualities = [] #initialise the kdtree qualities
        
        for index, individual in enumerate(current_population): #loop through the current population
            if not improvements: #if the improvements are not used
                current_fitness_score, pi_quality, kdtree_quality = fitness(individual, pi_digits) #identify the current fitness score, pi quality and kdtree quality
            else:
                current_fitness_score, pi_quality, kdtree_quality = fitness_improved(individual, pi_digits)
      
            
            fitness_score[index] = current_fitness_score #assign the current fitness score to the fitness score
            pi_qualities.append(pi_quality) #append the pi quality to the pi qualities
            kdtree_qualities.append(kdtree_quality) #append the kdtree quality to the kdtree qualities
        
        sorted_indices = sorted(fitness_score.keys(), key=lambda x: fitness_score[x], reverse=True) #sort the indices of the fitness score
        
        best_pi_quality_values.append(pi_qualities[sorted_indices[0]]) #append the best pi quality to the best pi quality values
        avg_pi_quality_values.append(sum(pi_qualities) / len(pi_qualities)) #append the average pi quality to the average pi quality values
        best_kdtree_values.append(kdtree_qualities[sorted_indices[0]]) #append the best kdtree quality to the best kdtree values
        avg_kdtree_values.append(sum(kdtree_qualities) / len(kdtree_qualities)) #append the average kdtree quality to the average kdtree values
        
        # Step 3.2: Select the best individuals for reproduction
        best_individuals = selection(current_population, sorted_indices) #identify the best individuals for reproduction
    
        # Step 3.3: Generate new offspring using crossover and mutation
        new_population = [] #initialise the new population
        new_population.append(current_population[sorted_indices[0]]) #append the current population to the new population
        new_population.append(current_population[sorted_indices[1]]) #append the current population to the new population
        while len(new_population) < population_size: #loop through the population size
            new_offspring_crossed = crossover(best_individuals[0], best_individuals[1], individual_length) #identify the new offspring crossed
            random_prob = np.random.rand() #identify the random probability
            if random_prob < mutation_rate: #if the random probability is less than the mutation rate
                new_offspring_crossed_mutated = mutation(new_offspring_crossed, individual_length, images, labels, kdtree_vector) #identify the new offspring crossed mutated
                new_population.append(new_offspring_crossed_mutated) #append the new offspring crossed mutated to the new population
            else: #if the random probability is greater than the mutation rate
                new_population.append(new_offspring_crossed) #append the new offspring crossed to the new population

   

        # Step 3.4: Replace the old population with the newly created individuals
        current_population = new_population #replace the old population with the newly created individuals


    # Step 4: Retrieve the best-performing individual from the final generation
    final_fitness_scores = {} #initialise the final fitness scores
    for index, individual in enumerate(current_population): #loop through the current population
        current_fitness_score, pi_quality, kdtree_quality = fitness(individual, pi_digits) #identify the current fitness score, pi quality and kdtree quality
        final_fitness_scores[index] = current_fitness_score #assign the current fitness score to the final fitness scores

    sorted_indices = sorted(final_fitness_scores.keys(), key=lambda x: final_fitness_scores[x], reverse=True) #sort the indices of the final fitness scores

    best_performing_individual = current_population[sorted_indices[0]] #identify the best performing individual

    # Step 5: Plot the fitness evolution
    plot_evolution(best_pi_quality_values, avg_pi_quality_values, best_kdtree_values, avg_kdtree_values) #plot the fitness evolution
    # Step 6: Plot the image output of the best individual
    plot_best_individual(best_performing_individual, pi_digits) #plot the image output of the best individual



# Main function to call the other functions
def main():
    ################### SESSION 1 ###################
    # Step 1: Download MNIST images
    # Call download_mnist_images() to download and save images
    images, labels = download_mnist_images()
    # Step 2: Analyze the first 5 images
    # Call analyze_images() for the first 5 images
    analyze_images(images[:5])
    # Step 3: Crop and resize the first 20 images
    # Call crop_image() for the first 20 images
    cropped_images = [] #keep track of the cropped images
    for i in range(20): #loop through the first 20 images
        img = images[i].reshape(28, 28) #reshape the image to 28x28
        cropped_img = crop_image(img) #crop the image
        cropped_images.append(cropped_img) #append the cropped image to the cropped_images list
    
    # Step 4: Display original vs cropped images (sample of 4)
    # Show comparison using matplotlib
    plt.figure(figsize=(10, 5)) #create the figure
    for i in range(4): #loop through the first 4 images
        # Original image
        plt.subplot(2, 4, i+1) #plot the original image
        plt.imshow(images[i].reshape(28, 28), cmap='gray') #plot the original image
        plt.title(f"Original {labels[i]}") #set the title of the original image
        plt.axis('off') #turn off the axis
        
        # Cropped image
        plt.subplot(2, 4, i+5) #plot the cropped image
        plt.imshow(cropped_images[i], cmap='gray') #plot the cropped image
        plt.title(f"Cropped {labels[i]}") #set the title of the cropped image
        plt.axis('off') #turn off the axis
    
    plt.tight_layout() #tight layout the plot
    plt.show() #show the plot
    
    # Step 5: Plot histogram comparison for cropped images
    # Call plot_histogram_comparison() for the cropped images
    plot_histogram_comparison(images, labels)


    ################### SESSION 2 ###################
    # Step 6: Generate KDTree vector
    kdtree_vector = classify_images_with_kdtree(images, labels) #generate the kdtree vector
    kdtree_vector_improved = classify_images_with_kdtree_improved(images, labels) #generate the kdtree vector with improvements
    
    # Step 7: Run GA 
    genetic_algorithm(images, labels, kdtree_vector) #run the genetic algorithm
    genetic_algorithm(images, labels, kdtree_vector_improved, improvements=True) #run the genetic algorithm with improvements


if __name__ == "__main__":
    main()


