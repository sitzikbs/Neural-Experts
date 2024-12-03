# a script that loads an image from sklearn and save it to the current directory as .png file

from skimage import data, io

# Dictionary of available images
available_images = {
    "camera": data.camera(),
    "astronaut": data.astronaut(),
    "checkerboard": data.checkerboard(),
    "coins": data.coins(),
    "horse": data.horse(),
    "clock": data.clock(),
    "page": data.page(),
    'cat': data.cat(),
    'coffee': data.coffee(),
    'rocket': data.rocket(),
}

# Display available images
print("Available images:")
for i, (name, _) in enumerate(available_images.items()):
    print(f"{i + 1}. {name}")

# Select an image
for im_name in available_images.keys():
    selected_image = available_images[im_name]
    # Save the selected image as a .png file
    io.imsave(f"{im_name}.png", selected_image)

    print(f"Image saved as {im_name}.png")