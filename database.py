import sqlite3
import os


connection = sqlite3.connect("labeled_images.db")
cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS labeled_images_binary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    label INTEGER NOT NULL
)
""")
connection.commit()
cursor.close()
connection.close()

connection = sqlite3.connect("labeled_images.db")
cursor = connection.cursor()

insert_query = "INSERT INTO labeled_images_binary (image, label) VALUES (?, ?)"

dataset_folder = "dataset/"

for label, folder_name in enumerate(["0", "1"]):
    folder_path = os.path.join(dataset_folder, folder_name)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        with open(image_path, "rb") as img_file:
            binary_data = img_file.read()
            cursor.execute(insert_query, (binary_data, label))

connection.commit()
cursor.close()
connection.close()

print("Image data inserted into SQLite!")
