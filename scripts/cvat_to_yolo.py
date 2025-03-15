import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

def convert_cvat_to_yolo(xml_file, output_dir, image_dir):
    """
    Convert CVAT XML annotations to YOLO format
    
    Args:
        xml_file: Path to CVAT XML file
        output_dir: Directory to save YOLO annotations
        image_dir: Directory containing the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get all labels/classes
    labels = []
    for label in root.findall('.//label'):
        labels.append(label.find('name').text)
    
    # Write classes.txt file
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"Found {len(labels)} classes: {', '.join(labels)}")
    
    # Process each image
    for image in tqdm(root.findall('.//image')):
        img_filename = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        
        # Create a TXT file for each image
        base_name = os.path.splitext(img_filename)[0]
        txt_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_file, 'w') as f:
            # Process each box/polygon/etc.
            for box in image.findall('.//box'):
                label = box.get('label')
                class_id = labels.index(label)
                
                # CVAT uses xmin, ymin, xmax, ymax
                xmin = float(box.get('xtl'))
                ymin = float(box.get('ytl'))
                xmax = float(box.get('xbr'))
                ymax = float(box.get('ybr'))
                
                # Convert to YOLO format: <class_id> <center_x> <center_y> <width> <height>
                # All values normalized to be between 0 and 1
                center_x = ((xmin + xmax) / 2) / img_width
                center_y = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
            
            # Process polygon annotations if they exist
            for polygon in image.findall('.//polygon'):
                label = polygon.get('label')
                class_id = labels.index(label)
                
                # For polygons, calculate bounding box
                points = polygon.get('points').split(';')
                x_coords = []
                y_coords = []
                
                for point in points:
                    x, y = map(float, point.split(','))
                    x_coords.append(x)
                    y_coords.append(y)
                
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)
                
                # Convert to YOLO format
                center_x = ((xmin + xmax) / 2) / img_width
                center_y = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CVAT XML to YOLO format')
    parser.add_argument('--xml', required=True, help='Path to CVAT XML file')
    parser.add_argument('--output', required=True, help='Output directory for YOLO annotations')
    parser.add_argument('--images', required=True, help='Directory containing the images')
    
    args = parser.parse_args()
    
    convert_cvat_to_yolo(args.xml, args.output, args.images)
    print("Conversion complete!")