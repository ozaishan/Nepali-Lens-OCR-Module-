input_path = "./dataset/labels.csv"
output_path = "./dataset/labels_comma.csv"

# Converting labels.csv format from [image_name text] => [image_name, text] (Added comma as seperator)

with open(input_path, "r", encoding="utf-16") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    outfile.write("image_file,text\n")  
    for line in infile:
        parts = line.strip().split(" ", 1)  # split only on first space
        if len(parts) == 2:
            image, text = parts
            outfile.write(f"{image},{text.strip()}\n")