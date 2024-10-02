import tensorflow_datasets as tfds

# List of categories you want to download
categories = ['classroom', 'restaurant', 'dining_room']

# Function to process a single category
def process_category(category):
    # Load the category
    ds, info = tfds.load(f'lsun/{category}', split='train', with_info=True, download=True)
    
    # Limit the dataset to 20,000 images
    limited_ds = ds.take(20000)
    
    # Count the number of images in the limited dataset
    count = limited_ds.reduce(0, lambda x, _: x + 1).numpy()
    print(f"Number of images in the limited '{category}' category: {count}")
    
    # Print dataset information
    print(f"Category: {category}")
    print(info)
    
    # Iterate over a few examples from the limited dataset
    print(f"Examples from category: {category}")
    for example in tfds.as_numpy(limited_ds.take(5)):
        print(example)

# Process each category one by one
for category in categories:
    process_category(category)