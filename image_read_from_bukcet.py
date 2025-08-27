import oci
from PIL import Image
from io import BytesIO

# Configuration file path
config = oci.config.from_file("config", "DEFAULT")

# Object Storage client
object_storage = oci.object_storage.ObjectStorageClient(config)

# Namespace and bucket details
namespace = object_storage.get_namespace().data
bucket_name = "your_bucket_name"

# List objects in the bucket
objects = object_storage.list_objects(namespace, bucket_name)
print("Objects in bucket:")
for obj in objects.data.objects:
    print(obj.name)

# Download a specific image (e.g., "example.jpg")
image_name = "example.jpg"  # Replace with the desired image name
response = object_storage.get_object(namespace, bucket_name, image_name)

# Load the image into a PIL Image object
image_data = response.data.content
image = Image.open(BytesIO(image_data))

# Display the image
image.show()

# Save the image locally if needed
image.save("downloaded_example.jpg")
print(f"Image {image_name} saved as 'downloaded_example.jpg'.")
