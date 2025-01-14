from PIL import Image, ImageDraw, ImageFont
import re
import os

data_out = "/Users/davidvandijcke/University of Michigan Dropbox/David Van Dijcke/rdd/data/out/"
sample_folder_path = data_out + 'frontline_9sep2024_selection'
sample_files = os.listdir(sample_folder_path)
sample_files.sort()

# Function to extract the week number from the filename
def extract_week_number(filename):
    match = re.search(r'isw_frontline_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Load all the images and their corresponding week numbers
images = []
for file in sample_files:
    if file.endswith('.png'):
        week_number = extract_week_number(file)
        if week_number:
            # Open the image
            img = Image.open(os.path.join(sample_folder_path, file)).convert("RGBA")
            
            # Add the week title to the image
            draw = ImageDraw.Draw(img)
            
            # Load a default font
            try:
                font = ImageFont.truetype("arial", 40)
            except IOError:
                font = ImageFont.load_default()
            
            # Define the text and its position
            text = f"Week {week_number}"
            text_position = (50, 50)
            
            # Add text to the image
            draw.text(text_position, text, font=font, fill=(255, 255, 255, 255))
            
            # Append the modified image
            images.append(img)

# Save the images as a GIF
gif_path = data_out + 'frontline_weekly_progress.gif'
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
