"""
General Utilities
"""
import datetime

def remove_extension(filename):
    last_index = filename.rfind(".")
    return filename[:last_index]

def generate_time_string():
    now = datetime.datetime.now()
    return now.strftime("%H%M%S")

def generate_date_string():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d")

def save_text(text, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")