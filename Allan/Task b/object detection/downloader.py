from bing_image_downloader import downloader

if __name__ == "__main__":
    # List of healthcare tools and objects categories
     categories = [
        "Stethoscope",
        "Ophthalmoscope",
        "Otoscope",
        "Blood Pressure Monitor",
        "Thermometer",
        "Scalpel",
        "Forceps",
        "Scissors",
        "Needle Holder",
        "Pulse Oximeter",
        "ECG Machine",
        "Holter Monitor",
        "Infusion Pump",
        "Ventilator",
        "Defibrillator",
        "X-ray Machine",
        "MRI",
        "CT Scan",
        "Microscope",
    ]

    # Set the output directory
    output_base_dir = "images/healthcare"

    # Initialize a variable to keep track of the total number of images downloaded
    total_images_downloaded = 0

    # Download 20 images for each category
    for category in categories:
        # Replace spaces with underscores in category names for folder names
        folder_name = category.replace(" ", "_")
        
        # Set the output directory for each category
        output_dir = f"{output_base_dir}/{folder_name}"
        
        # Download images for the current category
        print(f"Downloading images for {category}...")
        downloader.download(category, limit=20, output_dir=output_dir)
        
        # Reset the internal state to allow re-downloading for the next category
        downloader.reset()

        # Get the number of images downloaded for the current category
        images_downloaded = len(downloader.get_image_urls(category))
        total_images_downloaded += images_downloaded
        
        # Print the number of images downloaded for the current category
        print(f"Downloaded {images_downloaded} images for {category}\n")

    # Print the total number of images downloaded
    print(f"Total images downloaded: {total_images_downloaded}")
